# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core modules, which have been refactored in AlphaFold-Multimer.

The main difference is that MSA sampling pipeline is moved inside the JAX model
for easier implementation of recycling and ensembling.

Lower-level modules up to EvoformerIteration are reused from modules.py.
"""

import functools
from typing import Sequence

from alphafold.common import residue_constants
from alphafold.model import all_atom_multimer
from alphafold.model import common_modules
from alphafold.model import folding_multimer
from alphafold.model import geometry
from alphafold.model import layer_stack
from alphafold.model import modules
from alphafold.model import prng
from alphafold.model import utils

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def reduce_fn(x, mode):
  if mode == 'none' or mode is None:
    return jnp.asarray(x)
  elif mode == 'sum':
    return jnp.asarray(x).sum()
  elif mode == 'mean':
    return jnp.mean(jnp.asarray(x))
  else:
    raise ValueError('Unsupported reduction option.')


def gumbel_noise(key: jnp.ndarray, shape: Sequence[int]) -> jnp.ndarray:
  """Generate Gumbel Noise of given Shape.

  This generates samples from Gumbel(0, 1).

  Args:
    key: Jax random number key.
    shape: Shape of noise to return.

  Returns:
    Gumbel noise of given shape.
  """
  epsilon = 1e-6
  uniform = utils.padding_consistent_rng(jax.random.uniform)
  uniform_noise = uniform(
      key, shape=shape, dtype=jnp.float32, minval=0., maxval=1.)
  gumbel = -jnp.log(-jnp.log(uniform_noise + epsilon) + epsilon)
  return gumbel


def gumbel_max_sample(key: jnp.ndarray, logits: jnp.ndarray) -> jnp.ndarray:
  """Samples from a probability distribution given by 'logits'.

  This uses Gumbel-max trick to implement the sampling in an efficient manner.

  Args:
    key: prng key.
    logits: Logarithm of probabilities to sample from, probabilities can be
      unnormalized.

  Returns:
    Sample from logprobs in one-hot form.
  """
  z = gumbel_noise(key, logits.shape)
  return jax.nn.one_hot(
      jnp.argmax(logits + z, axis=-1),
      logits.shape[-1],
      dtype=logits.dtype)


def gumbel_argsort_sample_idx(key: jnp.ndarray,
                              logits: jnp.ndarray) -> jnp.ndarray:
  """Samples with replacement from a distribution given by 'logits'.

  This uses Gumbel trick to implement the sampling an efficient manner. For a
  distribution over k items this samples k times without replacement, so this
  is effectively sampling a random permutation with probabilities over the
  permutations derived from the logprobs.

  Args:
    key: prng key.
    logits: Logarithm of probabilities to sample from, probabilities can be
      unnormalized.

  Returns:
    Sample from logprobs in one-hot form.
  """
  z = gumbel_noise(key, logits.shape)
  # This construction is equivalent to jnp.argsort, but using a non stable sort,
  # since stable sort's aren't supported by jax2tf.
  axis = len(logits.shape) - 1
  iota = jax.lax.broadcasted_iota(jnp.int64, logits.shape, axis)
  _, perm = jax.lax.sort_key_val(
      logits + z, iota, dimension=-1, is_stable=False)
  return perm[::-1]


def make_masked_msa(batch, key, config, epsilon=1e-6):
  """Create data for BERT on raw MSA."""
  # Add a random amino acid uniformly.
  random_aa = jnp.array([0.05] * 20 + [0., 0.], dtype=jnp.float32)

  categorical_probs = (
      config.uniform_prob * random_aa +
      config.profile_prob * batch['msa_profile'] +
      config.same_prob * jax.nn.one_hot(batch['msa'], 22))

  # Put all remaining probability on [MASK] which is a new column.
  pad_shapes = [[0, 0] for _ in range(len(categorical_probs.shape))]
  pad_shapes[-1][1] = 1
  mask_prob = 1. - config.profile_prob - config.same_prob - config.uniform_prob
  assert mask_prob >= 0.
  categorical_probs = jnp.pad(
      categorical_probs, pad_shapes, constant_values=mask_prob)
  sh = batch['msa'].shape
  key, mask_subkey, gumbel_subkey = key.split(3)
  uniform = utils.padding_consistent_rng(jax.random.uniform)
  mask_position = uniform(mask_subkey.get(), sh) < config.replace_fraction
  mask_position *= batch['msa_mask']

  logits = jnp.log(categorical_probs + epsilon)
  bert_msa = gumbel_max_sample(gumbel_subkey.get(), logits)
  bert_msa = jnp.where(mask_position,
                       jnp.argmax(bert_msa, axis=-1), batch['msa'])
  bert_msa *= batch['msa_mask']

  # Mix real and masked MSA.
  if 'bert_mask' in batch:
    batch['bert_mask'] *= mask_position.astype(jnp.float32)
  else:
    batch['bert_mask'] = mask_position.astype(jnp.float32)
  batch['true_msa'] = batch['msa']
  batch['msa'] = bert_msa

  return batch


def nearest_neighbor_clusters(batch, gap_agreement_weight=0.):
  """Assign each extra MSA sequence to its nearest neighbor in sampled MSA."""

  # Determine how much weight we assign to each agreement.  In theory, we could
  # use a full blosum matrix here, but right now let's just down-weight gap
  # agreement because it could be spurious.
  # Never put weight on agreeing on BERT mask.

  weights = jnp.array(
      [1.] * 21 + [gap_agreement_weight] + [0.], dtype=jnp.float32)

  msa_mask = batch['msa_mask']
  msa_one_hot = jax.nn.one_hot(batch['msa'], 23)

  extra_mask = batch['extra_msa_mask']
  extra_one_hot = jax.nn.one_hot(batch['extra_msa'], 23)

  msa_one_hot_masked = msa_mask[:, :, None] * msa_one_hot
  extra_one_hot_masked = extra_mask[:, :, None] * extra_one_hot

  agreement = jnp.einsum('mrc, nrc->nm', extra_one_hot_masked,
                         weights * msa_one_hot_masked)

  cluster_assignment = jax.nn.softmax(1e3 * agreement, axis=0)
  cluster_assignment *= jnp.einsum('mr, nr->mn', msa_mask, extra_mask)

  cluster_count = jnp.sum(cluster_assignment, axis=-1)
  cluster_count += 1.  # We always include the sequence itself.

  msa_sum = jnp.einsum('nm, mrc->nrc', cluster_assignment, extra_one_hot_masked)
  msa_sum += msa_one_hot_masked

  cluster_profile = msa_sum / cluster_count[:, None, None]

  extra_deletion_matrix = batch['extra_deletion_matrix']
  deletion_matrix = batch['deletion_matrix']

  del_sum = jnp.einsum('nm, mc->nc', cluster_assignment,
                       extra_mask * extra_deletion_matrix)
  del_sum += deletion_matrix  # Original sequence.
  cluster_deletion_mean = del_sum / cluster_count[:, None]

  return cluster_profile, cluster_deletion_mean


def create_msa_feat(batch):
  """Create and concatenate MSA features."""
  msa_1hot = jax.nn.one_hot(batch['msa'], 23)
  deletion_matrix = batch['deletion_matrix']
  has_deletion = jnp.clip(deletion_matrix, 0., 1.)[..., None]
  deletion_value = (jnp.arctan(deletion_matrix / 3.) * (2. / jnp.pi))[..., None]

  deletion_mean_value = (jnp.arctan(batch['cluster_deletion_mean'] / 3.) *
                         (2. / jnp.pi))[..., None]

  msa_feat = [
      msa_1hot,
      has_deletion,
      deletion_value,
      batch['cluster_profile'],
      deletion_mean_value
  ]

  return jnp.concatenate(msa_feat, axis=-1)


def create_extra_msa_feature(batch, num_extra_msa):
  """Expand extra_msa into 1hot and concat with other extra msa features.

  We do this as late as possible as the one_hot extra msa can be very large.

  Args:
    batch: a dictionary with the following keys:
     * 'extra_msa': [num_seq, num_res] MSA that wasn't selected as a cluster
       centre. Note - This isn't one-hotted.
     * 'extra_deletion_matrix': [num_seq, num_res] Number of deletions at given
        position.
    num_extra_msa: Number of extra msa to use.

  Returns:
    Concatenated tensor of extra MSA features.
  """
  # 23 = 20 amino acids + 'X' for unknown + gap + bert mask
  extra_msa = batch['extra_msa'][:num_extra_msa]
  deletion_matrix = batch['extra_deletion_matrix'][:num_extra_msa]
  msa_1hot = jax.nn.one_hot(extra_msa, 23)
  has_deletion = jnp.clip(deletion_matrix, 0., 1.)[..., None]
  deletion_value = (jnp.arctan(deletion_matrix / 3.) * (2. / jnp.pi))[..., None]
  extra_msa_mask = batch['extra_msa_mask'][:num_extra_msa]
  return jnp.concatenate([msa_1hot, has_deletion, deletion_value],
                         axis=-1), extra_msa_mask


def sample_msa(key, batch, max_seq):
  """Sample MSA randomly, remaining sequences are stored as `extra_*`.

  Args:
    key: safe key for random number generation.
    batch: batch to sample msa from.
    max_seq: number of sequences to sample.
  Returns:
    Protein with sampled msa.
  """
  # Sample uniformly among sequences with at least one non-masked position.
  logits = (jnp.clip(jnp.sum(batch['msa_mask'], axis=-1), 0., 1.) - 1.) * 1e6
  # The cluster_bias_mask can be used to preserve the first row (target
  # sequence) for each chain, for example.
  if 'cluster_bias_mask' not in batch:
    cluster_bias_mask = jnp.pad(
        jnp.zeros(batch['msa'].shape[0] - 1), (1, 0), constant_values=1.)
  else:
    cluster_bias_mask = batch['cluster_bias_mask']

  logits += cluster_bias_mask * 1e6
  index_order = gumbel_argsort_sample_idx(key.get(), logits)
  sel_idx = index_order[:max_seq]
  extra_idx = index_order[max_seq:]

  for k in ['msa', 'deletion_matrix', 'msa_mask', 'bert_mask']:
    if k in batch:
      batch['extra_' + k] = batch[k][extra_idx]
      batch[k] = batch[k][sel_idx]

  return batch


def make_msa_profile(batch):
  """Compute the MSA profile."""

  # Compute the profile for every residue (over all MSA sequences).
  return utils.mask_mean(
      batch['msa_mask'][:, :, None], jax.nn.one_hot(batch['msa'], 22), axis=0)


class AlphaFoldIteration(hk.Module):
  """A single recycling iteration of AlphaFold architecture.

  Computes ensembled (averaged) representations from the provided features.
  These representations are then passed to the various heads
  that have been requested by the configuration file.
  """

  def __init__(self, config, global_config, name='alphafold_iteration'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,
               batch,
               is_training,
               return_representations=False,
               safe_key=None):

    if is_training:
      num_ensemble = np.asarray(self.config.num_ensemble_train)
    else:
      num_ensemble = np.asarray(self.config.num_ensemble_eval)

    # Compute representations for each MSA sample and average.
    embedding_module = EmbeddingsAndEvoformer(
        self.config.embeddings_and_evoformer, self.global_config)
    repr_shape = hk.eval_shape(
        lambda: embedding_module(batch, is_training))
    representations = {
        k: jnp.zeros(v.shape, v.dtype) for (k, v) in repr_shape.items()
    }

    def ensemble_body(x, unused_y):
      """Add into representations ensemble."""
      del unused_y
      representations, safe_key = x
      safe_key, safe_subkey = safe_key.split()
      representations_update = embedding_module(
          batch, is_training, safe_key=safe_subkey)

      for k in representations:
        if k not in {'msa', 'true_msa', 'bert_mask'}:
          representations[k] += representations_update[k] * (
              1. / num_ensemble).astype(representations[k].dtype)
        else:
          representations[k] = representations_update[k]

      return (representations, safe_key), None

    (representations, _), _ = hk.scan(
        ensemble_body, (representations, safe_key), None, length=num_ensemble)

    self.representations = representations
    self.batch = batch
    self.heads = {}
    for head_name, head_config in sorted(self.config.heads.items()):
      if not head_config.weight:
        continue  # Do not instantiate zero-weight heads.

      head_factory = {
          'masked_msa':
              modules.MaskedMsaHead,
          'distogram':
              modules.DistogramHead,
          'structure_module':
              folding_multimer.StructureModule,
          'predicted_aligned_error':
              modules.PredictedAlignedErrorHead,
          'predicted_lddt':
              modules.PredictedLDDTHead,
          'experimentally_resolved':
              modules.ExperimentallyResolvedHead,
      }[head_name]
      self.heads[head_name] = (head_config,
                               head_factory(head_config, self.global_config))

    structure_module_output = None
    if 'entity_id' in batch and 'all_atom_positions' in batch:
      _, fold_module = self.heads['structure_module']
      structure_module_output = fold_module(representations, batch, is_training)

    ret = {}
    ret['representations'] = representations

    for name, (head_config, module) in self.heads.items():
      if name == 'structure_module' and structure_module_output is not None:
        ret[name] = structure_module_output
        representations['structure_module'] = structure_module_output.pop('act')
      # Skip confidence heads until StructureModule is executed.
      elif name in {'predicted_lddt', 'predicted_aligned_error',
                    'experimentally_resolved'}:
        continue
      else:
        ret[name] = module(representations, batch, is_training)

    # Add confidence heads after StructureModule is executed.
    if self.config.heads.get('predicted_lddt.weight', 0.0):
      name = 'predicted_lddt'
      head_config, module = self.heads[name]
      ret[name] = module(representations, batch, is_training)

    if self.config.heads.experimentally_resolved.weight:
      name = 'experimentally_resolved'
      head_config, module = self.heads[name]
      ret[name] = module(representations, batch, is_training)

    if self.config.heads.get('predicted_aligned_error.weight', 0.0):
      name = 'predicted_aligned_error'
      head_config, module = self.heads[name]
      ret[name] = module(representations, batch, is_training)
      # Will be used for ipTM computation.
      ret[name]['asym_id'] = batch['asym_id']

    return ret


class AlphaFold(hk.Module):
  """AlphaFold-Multimer model with recycling.
  """

  def __init__(self, config, name='alphafold'):
    super().__init__(name=name)
    self.config = config
    self.global_config = config.global_config

  def __call__(
      self,
      batch,
      is_training,
      return_representations=False,
      safe_key=None):

    c = self.config
    impl = AlphaFoldIteration(c, self.global_config)

    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())
    elif isinstance(safe_key, jnp.ndarray):
      safe_key = prng.SafeKey(safe_key)

    assert isinstance(batch, dict)
    num_res = batch['aatype'].shape[0]

    def get_prev(ret):
      new_prev = {
          'prev_pos':
              ret['structure_module']['final_atom_positions'],
          'prev_msa_first_row': ret['representations']['msa_first_row'],
          'prev_pair': ret['representations']['pair'],
      }
      return jax.tree_map(jax.lax.stop_gradient, new_prev)

    def apply_network(prev, safe_key):
      recycled_batch = {**batch, **prev}
      return impl(
          batch=recycled_batch,
          is_training=is_training,
          safe_key=safe_key)

    prev = {}
    emb_config = self.config.embeddings_and_evoformer
    if emb_config.recycle_pos:
      prev['prev_pos'] = jnp.zeros(
          [num_res, residue_constants.atom_type_num, 3])
    if emb_config.recycle_features:
      prev['prev_msa_first_row'] = jnp.zeros(
          [num_res, emb_config.msa_channel])
      prev['prev_pair'] = jnp.zeros(
          [num_res, num_res, emb_config.pair_channel])

    if self.config.num_recycle:
      if 'num_iter_recycling' in batch:
        # Training time: num_iter_recycling is in batch.
        # Value for each ensemble batch is the same, so arbitrarily taking 0-th.
        num_iter = batch['num_iter_recycling'][0]

        # Add insurance that even when ensembling, we will not run more
        # recyclings than the model is configured to run.
        num_iter = jnp.minimum(num_iter, c.num_recycle)
      else:
        # Eval mode or tests: use the maximum number of iterations.
        num_iter = c.num_recycle

      def distances(points):
        """Compute all pairwise distances for a set of points."""
        return jnp.sqrt(jnp.sum((points[:, None] - points[None, :])**2,
                                axis=-1))

      def recycle_body(x):
        i, _, prev, safe_key = x
        safe_key1, safe_key2 = safe_key.split() if c.resample_msa_in_recycling else safe_key.duplicate()  # pylint: disable=line-too-long
        ret = apply_network(prev=prev, safe_key=safe_key2)
        return i+1, prev, get_prev(ret), safe_key1

      def recycle_cond(x):
        i, prev, next_in, _ = x
        ca_idx = residue_constants.atom_order['CA']
        sq_diff = jnp.square(distances(prev['prev_pos'][:, ca_idx, :]) -
                             distances(next_in['prev_pos'][:, ca_idx, :]))
        mask = batch['seq_mask'][:, None] * batch['seq_mask'][None, :]
        sq_diff = utils.mask_mean(mask, sq_diff)
        # Early stopping criteria based on criteria used in
        # AF2Complex: https://www.nature.com/articles/s41467-022-29394-2
        diff = jnp.sqrt(sq_diff + 1e-8)  # avoid bad numerics giving negatives
        less_than_max_recycles = (i < num_iter)
        has_exceeded_tolerance = (
            (i == 0) | (diff > c.recycle_early_stop_tolerance))
        return less_than_max_recycles & has_exceeded_tolerance

      if hk.running_init():
        num_recycles, _, prev, safe_key = recycle_body(
            (0, prev, prev, safe_key))
      else:
        num_recycles, _, prev, safe_key = hk.while_loop(
            recycle_cond,
            recycle_body,
            (0, prev, prev, safe_key))
    else:
      # No recycling.
      num_recycles = 0

    # Run extra iteration.
    ret = apply_network(prev=prev, safe_key=safe_key)

    if not return_representations:
      del ret['representations']
    ret['num_recycles'] = num_recycles

    return ret


class EmbeddingsAndEvoformer(hk.Module):
  """Embeds the input data and runs Evoformer."""

  def __init__(self, config, global_config, name='evoformer'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config
    self.all_outputs = {}

  def save_outputs(self, outputs, num_blocks):
    """Save outputs to a file."""
    import json
    import os
    
    filename = f'evoformer_outputs_{num_blocks}_blocks.json'
    with open(filename, 'w') as f:
      json.dump(outputs, f, indent=2)
    print(f"Saved outputs to {filename}")

  def run_evoformer_stack(self, evoformer_input, masks, num_blocks, is_training, safe_key):
    """Run Evoformer stack with specified number of blocks."""
    c = self.config
    gc = self.global_config

    evoformer_iteration = EvoformerIteration(
        c.evoformer, gc, is_extra_msa=False, name=f'evoformer_iteration_{num_blocks}')

    def evoformer_fn(x):
      act, safe_key, iteration = x
      safe_key, safe_subkey = safe_key.split()
      evoformer_output = evoformer_iteration(
          activations=act,
          masks=masks,
          iteration=iteration,
          is_training=is_training,
          safe_key=safe_subkey)
      return (evoformer_output, safe_key, iteration + 1)

    if gc.use_remat:
      evoformer_fn = hk.remat(evoformer_fn)

    safe_key, safe_subkey = safe_key.split()
    evoformer_stack = layer_stack.layer_stack(num_blocks)(evoformer_fn)
    
    # Run the stack
    (evoformer_output, _, _), _ = evoformer_stack(
        (evoformer_input, safe_subkey, 0), None)

    # Save the outputs
    self.save_outputs(evoformer_iteration.layer_outputs, num_blocks)
    
    return evoformer_output

  def __call__(self, batch, is_training, safe_key=None):
    """Embeds the input data and runs Evoformer."""
    c = self.config
    gc = self.global_config

    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())

    # Initial embeddings
    preprocess_1d = common_modules.Linear(
        c.msa_channel, name='preprocess_1d')(
            batch['target_feat'])

    preprocess_msa = common_modules.Linear(
        c.msa_channel, name='preprocess_msa')(
            batch['msa_feat'])

    msa_activations = jnp.expand_dims(preprocess_1d, axis=0) + preprocess_msa

    left_single = common_modules.Linear(
        c.pair_channel, name='left_single')(
            batch['target_feat'])
    right_single = common_modules.Linear(
        c.pair_channel, name='right_single')(
            batch['target_feat'])
    pair_activations = left_single[:, None] + right_single[None]
    mask_2d = batch['seq_mask'][:, None] * batch['seq_mask'][None, :]

    # Process MSA features
    msa_mask = batch['msa_mask']
    
    # Run with 5 blocks
    evoformer_input_5 = {
        'msa': msa_activations,
        'pair': pair_activations,
    }
    evoformer_masks = {
        'msa': msa_mask,
        'pair': mask_2d
    }
    
    safe_key, safe_subkey = safe_key.split()
    output_5 = self.run_evoformer_stack(
        evoformer_input_5, evoformer_masks, 5, is_training, safe_subkey)
    
    # Run with 48 blocks
    safe_key, safe_subkey = safe_key.split()
    output_48 = self.run_evoformer_stack(
        evoformer_input_5, evoformer_masks, 48, is_training, safe_subkey)
    
    # Compare and save the difference
    diff_msa = jnp.mean(jnp.abs(output_48['msa'] - output_5['msa']))
    diff_pair = jnp.mean(jnp.abs(output_48['pair'] - output_5['pair']))
    
    comparison = {
        'msa_difference_mean': float(diff_msa),
        'pair_difference_mean': float(diff_pair)
    }
    
    with open('evoformer_comparison.json', 'w') as f:
      json.dump(comparison, f, indent=2)
    print("Saved comparison to evoformer_comparison.json")
    
    # Return the 48-block output as the final result
    return output_48


class TemplateEmbedding(hk.Module):
  """Embed a set of templates."""

  def __init__(self, config, global_config, name='template_embedding'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, query_embedding, template_batch, padding_mask_2d,
               multichain_mask_2d, is_training,
               safe_key=None):
    """Generate an embedding for a set of templates.

    Args:
      query_embedding: [num_res, num_res, num_channel] a query tensor that will
        be used to attend over the templates to remove the num_templates
        dimension.
      template_batch: A dictionary containing:
        `template_aatype`: [num_templates, num_res] aatype for each template.
        `template_all_atom_positions`: [num_templates, num_res, 37, 3] atom
          positions for all templates.
        `template_all_atom_mask`: [num_templates, num_res, 37] mask for each
          template.
      padding_mask_2d: [num_res, num_res] Pair mask for attention operations.
      multichain_mask_2d: [num_res, num_res] Mask indicating which residue pairs
        are intra-chain, used to mask out residue distance based features
        between chains.
      is_training: bool indicating where we are running in training mode.
      safe_key: random key generator.

    Returns:
      An embedding of size [num_res, num_res, num_channels]
    """
    c = self.config
    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())

    num_templates = template_batch['template_aatype'].shape[0]
    num_res, _, query_num_channels = query_embedding.shape

    # Embed each template separately.
    template_embedder = SingleTemplateEmbedding(self.config, self.global_config)
    def partial_template_embedder(template_aatype,
                                  template_all_atom_positions,
                                  template_all_atom_mask,
                                  unsafe_key):
      safe_key = prng.SafeKey(unsafe_key)
      return template_embedder(query_embedding,
                               template_aatype,
                               template_all_atom_positions,
                               template_all_atom_mask,
                               padding_mask_2d,
                               multichain_mask_2d,
                               is_training,
                               safe_key)

    safe_key, unsafe_key = safe_key.split()
    unsafe_keys = jax.random.split(unsafe_key._key, num_templates)

    def scan_fn(carry, x):
      return carry + partial_template_embedder(*x), None

    scan_init = jnp.zeros((num_res, num_res, c.num_channels),
                          dtype=query_embedding.dtype)
    summed_template_embeddings, _ = hk.scan(
        scan_fn, scan_init,
        (template_batch['template_aatype'],
         template_batch['template_all_atom_positions'],
         template_batch['template_all_atom_mask'], unsafe_keys))

    embedding = summed_template_embeddings / num_templates
    embedding = jax.nn.relu(embedding)
    embedding = common_modules.Linear(
        query_num_channels,
        initializer='relu',
        name='output_linear')(embedding)

    return embedding


class SingleTemplateEmbedding(hk.Module):
  """Embed a single template."""

  def __init__(self, config, global_config, name='single_template_embedding'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, query_embedding, template_aatype,
               template_all_atom_positions, template_all_atom_mask,
               padding_mask_2d, multichain_mask_2d, is_training,
               safe_key):
    """Build the single template embedding graph.

    Args:
      query_embedding: (num_res, num_res, num_channels) - embedding of the
        query sequence/msa.
      template_aatype: [num_res] aatype for each template.
      template_all_atom_positions: [num_res, 37, 3] atom positions for all
        templates.
      template_all_atom_mask: [num_res, 37] mask for each template.
      padding_mask_2d: Padding mask (Note: this doesn't care if a template
        exists, unlike the template_pseudo_beta_mask).
      multichain_mask_2d: A mask indicating intra-chain residue pairs, used
        to mask out between chain distances/features when templates are for
        single chains.
      is_training: Are we in training mode.
      safe_key: Random key generator.

    Returns:
      A template embedding (num_res, num_res, num_channels).
    """
    gc = self.global_config
    c = self.config
    assert padding_mask_2d.dtype == query_embedding.dtype
    dtype = query_embedding.dtype
    num_channels = self.config.num_channels

    def construct_input(query_embedding, template_aatype,
                        template_all_atom_positions, template_all_atom_mask,
                        multichain_mask_2d):

      # Compute distogram feature for the template.
      template_positions, pseudo_beta_mask = modules.pseudo_beta_fn(
          template_aatype, template_all_atom_positions, template_all_atom_mask)
      pseudo_beta_mask_2d = (pseudo_beta_mask[:, None] *
                             pseudo_beta_mask[None, :])
      pseudo_beta_mask_2d *= multichain_mask_2d
      template_dgram = modules.dgram_from_positions(
          template_positions, **self.config.dgram_features)
      template_dgram *= pseudo_beta_mask_2d[..., None]
      template_dgram = template_dgram.astype(dtype)
      pseudo_beta_mask_2d = pseudo_beta_mask_2d.astype(dtype)
      to_concat = [(template_dgram, 1), (pseudo_beta_mask_2d, 0)]

      aatype = jax.nn.one_hot(template_aatype, 22, axis=-1, dtype=dtype)
      to_concat.append((aatype[None, :, :], 1))
      to_concat.append((aatype[:, None, :], 1))

      # Compute a feature representing the normalized vector between each
      # backbone affine - i.e. in each residues local frame, what direction are
      # each of the other residues.
      raw_atom_pos = template_all_atom_positions
      if gc.bfloat16:
        # Vec3Arrays are required to be float32
        raw_atom_pos = raw_atom_pos.astype(jnp.float32)

      atom_pos = geometry.Vec3Array.from_array(raw_atom_pos)
      rigid, backbone_mask = folding_multimer.make_backbone_affine(
          atom_pos,
          template_all_atom_mask,
          template_aatype)
      points = rigid.translation
      rigid_vec = rigid[:, None].inverse().apply_to_point(points)
      unit_vector = rigid_vec.normalized()
      unit_vector = [unit_vector.x, unit_vector.y, unit_vector.z]

      if gc.bfloat16:
        unit_vector = [x.astype(jnp.bfloat16) for x in unit_vector]
        backbone_mask = backbone_mask.astype(jnp.bfloat16)

      backbone_mask_2d = backbone_mask[:, None] * backbone_mask[None, :]
      backbone_mask_2d *= multichain_mask_2d
      unit_vector = [x*backbone_mask_2d for x in unit_vector]

      # Note that the backbone_mask takes into account C, CA and N (unlike
      # pseudo beta mask which just needs CB) so we add both masks as features.
      to_concat.extend([(x, 0) for x in unit_vector])
      to_concat.append((backbone_mask_2d, 0))

      query_embedding = common_modules.LayerNorm(
          axis=[-1],
          create_scale=True,
          create_offset=True,
          name='query_embedding_norm')(
              query_embedding)
      # Allow the template embedder to see the query embedding.  Note this
      # contains the position relative feature, so this is how the network knows
      # which residues are next to each other.
      to_concat.append((query_embedding, 1))

      act = 0

      for i, (x, n_input_dims) in enumerate(to_concat):

        act += common_modules.Linear(
            num_channels,
            num_input_dims=n_input_dims,
            initializer='relu',
            name=f'template_pair_embedding_{i}')(x)
      return act

    act = construct_input(query_embedding, template_aatype,
                          template_all_atom_positions, template_all_atom_mask,
                          multichain_mask_2d)

    template_iteration = TemplateEmbeddingIteration(
        c.template_pair_stack, gc, name='template_embedding_iteration')

    def template_iteration_fn(x):
      act, safe_key = x

      safe_key, safe_subkey = safe_key.split()
      act = template_iteration(
          act=act,
          pair_mask=padding_mask_2d,
          is_training=is_training,
          safe_key=safe_subkey)
      return (act, safe_key)

    if gc.use_remat:
      template_iteration_fn = hk.remat(template_iteration_fn)

    safe_key, safe_subkey = safe_key.split()
    template_stack = layer_stack.layer_stack(
        c.template_pair_stack.num_block)(
            template_iteration_fn)
    act, safe_key = template_stack((act, safe_subkey))

    act = common_modules.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='output_layer_norm')(
            act)

    return act


class TemplateEmbeddingIteration(hk.Module):
  """Single Iteration of Template Embedding."""

  def __init__(self, config, global_config,
               name='template_embedding_iteration'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, act, pair_mask, is_training=True,
               safe_key=None):
    """Build a single iteration of the template embedder.

    Args:
      act: [num_res, num_res, num_channel] Input pairwise activations.
      pair_mask: [num_res, num_res] padding mask.
      is_training: Whether to run in training mode.
      safe_key: Safe pseudo-random generator key.

    Returns:
      [num_res, num_res, num_channel] tensor of activations.
    """
    c = self.config
    gc = self.global_config

    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())

    dropout_wrapper_fn = functools.partial(
        modules.dropout_wrapper,
        is_training=is_training,
        global_config=gc)

    safe_key, *sub_keys = safe_key.split(20)
    sub_keys = iter(sub_keys)

    act = dropout_wrapper_fn(
        modules.TriangleMultiplication(c.triangle_multiplication_outgoing, gc,
                                       name='triangle_multiplication_outgoing'),
        act,
        pair_mask,
        safe_key=next(sub_keys))

    act = dropout_wrapper_fn(
        modules.TriangleMultiplication(c.triangle_multiplication_incoming, gc,
                                       name='triangle_multiplication_incoming'),
        act,
        pair_mask,
        safe_key=next(sub_keys))
    act = dropout_wrapper_fn(
        modules.TriangleAttention(c.triangle_attention_starting_node, gc,
                                  name='triangle_attention_starting_node'),
        act,
        pair_mask,
        safe_key=next(sub_keys))
    act = dropout_wrapper_fn(
        modules.TriangleAttention(c.triangle_attention_ending_node, gc,
                                  name='triangle_attention_ending_node'),
        act,
        pair_mask,
        safe_key=next(sub_keys))
    act = dropout_wrapper_fn(
        modules.Transition(c.pair_transition, gc,
                           name='pair_transition'),
        act,
        pair_mask,
        safe_key=next(sub_keys))

    return act


def template_embedding_1d(batch, num_channel, global_config):
  """Embed templates into an (num_res, num_templates, num_channels) embedding.

  Args:
    batch: A batch containing:
      template_aatype, (num_templates, num_res) aatype for the templates.
      template_all_atom_positions, (num_templates, num_residues, 37, 3) atom
        positions for the templates.
      template_all_atom_mask, (num_templates, num_residues, 37) atom mask for
        each template.
    num_channel: The number of channels in the output.
    global_config: The global_config.

  Returns:
    An embedding of shape (num_templates, num_res, num_channels) and a mask of
    shape (num_templates, num_res).
  """

  # Embed the templates aatypes.
  aatype_one_hot = jax.nn.one_hot(batch['template_aatype'], 22, axis=-1)

  num_templates = batch['template_aatype'].shape[0]
  all_chi_angles = []
  all_chi_masks = []
  for i in range(num_templates):
    atom_pos = geometry.Vec3Array.from_array(
        batch['template_all_atom_positions'][i, :, :, :])
    template_chi_angles, template_chi_mask = all_atom_multimer.compute_chi_angles(
        atom_pos,
        batch['template_all_atom_mask'][i, :, :],
        batch['template_aatype'][i, :])
    all_chi_angles.append(template_chi_angles)
    all_chi_masks.append(template_chi_mask)
  chi_angles = jnp.stack(all_chi_angles, axis=0)
  chi_mask = jnp.stack(all_chi_masks, axis=0)

  template_features = jnp.concatenate([
      aatype_one_hot,
      jnp.sin(chi_angles) * chi_mask,
      jnp.cos(chi_angles) * chi_mask,
      chi_mask], axis=-1)

  template_mask = chi_mask[:, :, 0]

  if global_config.bfloat16:
    template_features = template_features.astype(jnp.bfloat16)
    template_mask = template_mask.astype(jnp.bfloat16)

  template_activations = common_modules.Linear(
      num_channel,
      initializer='relu',
      name='template_single_embedding')(
          template_features)
  template_activations = jax.nn.relu(template_activations)
  template_activations = common_modules.Linear(
      num_channel,
      initializer='relu',
      name='template_projection')(
          template_activations)
  return template_activations, template_mask


class EvoformerIteration(hk.Module):
  """Single iteration (block) of Evoformer stack."""

  def __init__(self, config, global_config, is_extra_msa, name='evoformer_iteration'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config
    self.is_extra_msa = is_extra_msa
    self.layer_outputs = []

  def save_layer_output(self, pair_act, layer_name, iteration):
    """Save the output of each layer for analysis."""
    # Calculate some meaningful metrics from pair_act
    avg_value = jnp.mean(pair_act)
    max_value = jnp.max(pair_act)
    min_value = jnp.min(pair_act)
    
    output = {
        'layer_name': layer_name,
        'iteration': iteration,
        'avg_value': float(avg_value),
        'max_value': float(max_value),
        'min_value': float(min_value)
    }
    
    self.layer_outputs.append(output)

  def __call__(self, activations, masks, iteration=0, is_training=True, safe_key=None):
    """Builds EvoformerIteration module."""
    c = self.config
    gc = self.global_config

    msa_act, pair_act = activations['msa'], activations['pair']

    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())

    msa_mask, pair_mask = masks['msa'], masks['pair']

    dropout_wrapper_fn = functools.partial(
        modules.dropout_wrapper,
        is_training=is_training,
        global_config=gc)

    safe_key, *sub_keys = safe_key.split(10)
    sub_keys = iter(sub_keys)

    self.save_layer_output(pair_act, 'input', iteration)

    outer_module = modules.OuterProductMean(
        config=c.outer_product_mean,
        global_config=self.global_config,
        num_output_channel=int(pair_act.shape[-1]),
        name='outer_product_mean')
    if c.outer_product_mean.first:
      pair_act = dropout_wrapper_fn(
          outer_module,
          msa_act,
          msa_mask,
          safe_key=next(sub_keys),
          output_act=pair_act)
      self.save_layer_output(pair_act, 'outer_product_mean', iteration)

    msa_act = dropout_wrapper_fn(
        modules.MSARowAttentionWithPairBias(
            c.msa_row_attention_with_pair_bias, gc,
            name='msa_row_attention_with_pair_bias'),
        msa_act,
        msa_mask,
        safe_key=next(sub_keys),
        pair_act=pair_act)
    self.save_layer_output(pair_act, 'msa_row_attention', iteration)

    if not self.is_extra_msa:
      attn_mod = modules.MSAColumnAttention(
          c.msa_column_attention, gc, name='msa_column_attention')
    else:
      attn_mod = modules.MSAColumnGlobalAttention(
          c.msa_column_attention, gc, name='msa_column_global_attention')
    msa_act = dropout_wrapper_fn(
        attn_mod,
        msa_act,
        msa_mask,
        safe_key=next(sub_keys))
    self.save_layer_output(pair_act, 'msa_column_attention', iteration)

    msa_act = dropout_wrapper_fn(
        modules.Transition(c.msa_transition, gc, name='msa_transition'),
        msa_act,
        msa_mask,
        safe_key=next(sub_keys))
    self.save_layer_output(pair_act, 'msa_transition', iteration)

    if not c.outer_product_mean.first:
      pair_act = dropout_wrapper_fn(
          outer_module,
          msa_act,
          msa_mask,
          safe_key=next(sub_keys),
          output_act=pair_act)
      self.save_layer_output(pair_act, 'outer_product_mean_2', iteration)

    pair_act = dropout_wrapper_fn(
        modules.TriangleMultiplication(c.triangle_multiplication_outgoing, gc,
                                   name='triangle_multiplication_outgoing'),
        pair_act,
        pair_mask,
        safe_key=next(sub_keys))
    self.save_layer_output(pair_act, 'triangle_multiplication_outgoing', iteration)

    pair_act = dropout_wrapper_fn(
        modules.TriangleMultiplication(c.triangle_multiplication_incoming, gc,
                                   name='triangle_multiplication_incoming'),
        pair_act,
        pair_mask,
        safe_key=next(sub_keys))
    self.save_layer_output(pair_act, 'triangle_multiplication_incoming', iteration)

    pair_act = dropout_wrapper_fn(
        modules.TriangleAttention(c.triangle_attention_starting_node, gc,
                              name='triangle_attention_starting_node'),
        pair_act,
        pair_mask,
        safe_key=next(sub_keys))
    self.save_layer_output(pair_act, 'triangle_attention_starting', iteration)

    pair_act = dropout_wrapper_fn(
        modules.TriangleAttention(c.triangle_attention_ending_node, gc,
                              name='triangle_attention_ending_node'),
        pair_act,
        pair_mask,
        safe_key=next(sub_keys))
    self.save_layer_output(pair_act, 'triangle_attention_ending', iteration)

    pair_act = dropout_wrapper_fn(
        modules.Transition(c.pair_transition, gc, name='pair_transition'),
        pair_act,
        pair_mask,
        safe_key=next(sub_keys))
    self.save_layer_output(pair_act, 'pair_transition', iteration)

    return {'msa': msa_act, 'pair': pair_act}
