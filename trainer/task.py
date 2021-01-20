import argparse
def get_args():
  args_parser = argparse.ArgumentParser()

  args_parser.add_argument(
    '--data-path',
    required=True,
  )
  args_parser.add_argument(
    '--bucket-name',
    required=True,
  )

  # Train arguments
  args_parser.add_argument(
    '--seed',
    default=42,
    type=int,
  )
  args_parser.add_argument(
    '--epochs',
    default=20,
    type=int,
  )
  args_parser.add_argument(
    '--batch-size',
    default=10,
    type=int,
  )
  args_parser.add_argument(
    '--learning-rate',
    default=0.0001,
    type=float,
  )
  args_parser.add_argument(
    '--weight-decay',
    default=0,
    type=float,
  )
  args_parser.add_argument(
    '--beta1',
    default=0.9,
    type=float,
  )
  args_parser.add_argument(
    '--beta2',
    default=0.999,
    type=float,
  )
  
  return args_parser.parse_args()

import mnist
if __name__ == '__main__':
  args = get_args()
  mnist.run(**vars(args))
