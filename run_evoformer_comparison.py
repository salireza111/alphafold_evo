import json
import jax
import haiku as hk
from alphafold.model import modules_multimer
from alphafold.model import config

def main():
    """Run Evoformer comparison between 5 and 48 iterations."""
    
    # Create a dummy batch for testing
    batch_size = 1
    num_res = 10
    num_msa = 5
    
    batch = {
        'target_feat': jax.random.normal(jax.random.PRNGKey(0), (num_res, 21)),
        'msa_feat': jax.random.normal(jax.random.PRNGKey(1), (num_msa, num_res, 49)),
        'msa_mask': jax.numpy.ones((num_msa, num_res)),
        'seq_mask': jax.numpy.ones(num_res),
    }
    
    # Get default config
    model_config = config.model_config('model_1_ptm')
    
    # Create the model
    def forward_fn(batch):
        model = modules_multimer.EmbeddingsAndEvoformer(
            model_config.model.embeddings_and_evoformer,
            model_config.model.global_config)
        return model(batch, is_training=True)
    
    # Transform into a pure function
    forward = hk.transform(forward_fn)
    
    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    params = forward.init(rng, batch)
    
    # Run forward pass
    output = forward.apply(params, rng, batch)
    
    print("\nComparison complete! Results have been saved to:")
    print("1. evoformer_outputs_5_blocks.json")
    print("2. evoformer_outputs_48_blocks.json")
    print("3. evoformer_comparison.json")
    
    # Print summary of differences
    with open('evoformer_comparison.json', 'r') as f:
        comparison = json.load(f)
    print("\nDifferences between 5 and 48 iterations:")
    print(f"MSA mean difference: {comparison['msa_difference_mean']:.6f}")
    print(f"Pair mean difference: {comparison['pair_difference_mean']:.6f}")

if __name__ == '__main__':
    main() 