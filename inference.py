import argparse

import torch
from model import Generator, RNN


def generate_text(checkpoint_path, initial_str, how_many=100, temperature=0.85, max_length=10):
    """
    Generate text using a pre-trained RNN model.

    Parameters:
        checkpoint_path (str): Path to the checkpoint file.
        initial_str (str): Initial string to start the generation.
        how_many (int): Number of characters to generate.
        temperature (float): Temperature parameter for controlling the randomness of the generation.
        max_length (int): Maximum length of the generated text.

    Returns:
        str: Generated text.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path)
    model = RNN(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        output_size=checkpoint['output_size']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    gen = Generator(model)
    generated_text = gen.generate(initial_str=initial_str, how_many=how_many, temperature=temperature,
                                  max_length=max_length)
    return generated_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text using a pre-trained RNN model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_epoch_344_1.6779072265625.pth',
                        help='Path to the checkpoint file')
    parser.add_argument('--initial_str', type=str, default='Am',
                        help='Initial string to start the generation')
    parser.add_argument('--how_many', type=int, default=100,
                        help='Number of characters to generate')
    parser.add_argument('--temperature', type=float, default=0.85,
                        help='Temperature parameter for controlling the randomness of the generation')
    parser.add_argument('--max_length', type=int, default=10,
                        help='Maximum length of the generated text')
    args = parser.parse_args()

    generated_text = generate_text(args.checkpoint, args.initial_str, args.how_many, args.temperature, args.max_length)
    print(generated_text)
