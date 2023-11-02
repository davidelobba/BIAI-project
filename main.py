import argparse
import ga
import pso
import cma_es
import cppn
import os

def main():
    parser = argparse.ArgumentParser(description='Run evolutionary algorithms.')
    parser.add_argument('--algorithm', type=str, choices=['ga', 'pso', 'cma_es', 'cppn'], default='ga', help='Algorithm to run')
    parser.add_argument('--config-path', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--weights-path', type=str, default=None, help='Path to weights file')
    parser.add_argument('--network', type=str, choices=['resnet18', 'resnet34', 'resnet50'], default='resnet18', help='Name of the network')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'imagenet', 'mnist'], default='cifar10', help='Name of the dataset')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to save the best individual')
    parser.add_argument('--test-model', dest='test', action='store_true', help='Test the model before running the algorithm')
    parser.add_argument('--no-test-model', dest='test', action='store_false', help='Do not test the model before running the algorithm')
    parser.set_defaults(test=False)
    parser.add_argument('--wandb', dest='wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false', help='Do not use wandb for logging')
    parser.set_defaults(wandb=False)
    parser.add_argument('--save', dest='save', action='store_true', help='Save the best individual')
    parser.add_argument('--no-save', dest='save', action='store_false', help='Do not save the best individual')
    parser.set_defaults(save=False)
    parser.add_argument('--adaptive-mutation-crossover', dest='adaptive_mutation_crossover', action='store_true', help='Use adaptive mutation and crossover')
    parser.add_argument('--normalize', dest='normalize', action='store_true', help='Use normalization')
    parser.add_argument('--class-constraint', type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], default=None, help='Class constraint for the algorithm' )
    
    args = parser.parse_args()
    if args.dataset != 'imagenet':
        weights_path = os.path.join(args.weights_path, f'{args.dataset}', f'{args.network}_best_model.pth')
    else:
        weights_path = args.weights_path

    print(f"Running {args.algorithm} on {args.dataset} with {args.network} network, weights from {weights_path}")

    if args.algorithm == 'ga':
        args.output_dir = os.path.join(args.output_dir, 'ga')
        ga.run_ga(args, weights_path)

    elif args.algorithm == 'pso':
        args.output_dir = os.path.join(args.output_dir, 'pso')
        pso.run_pso(args, weights_path)
    
    elif args.algorithm == 'cma_es':
        args.output_dir = os.path.join(args.output_dir, 'cma_es')
        cma_es.run_cma_es(args, weights_path)
    
    elif args.algorithm == 'cppn':
        args.output_dir = os.path.join(args.output_dir, 'cppn')
        cppn.run_cppn(args, weights_path)

if __name__ == "__main__":
    main()
