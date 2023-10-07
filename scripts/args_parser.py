import yaml
import argparse
from transformers import TrainingArguments

class FileLoader:
    @staticmethod
    def load_from_yamlfile(filename):
        try:
            with open(filename, 'r') as file:
                config = yaml.safe_load(file)
                print("Loaded config from yaml file:", config)
                return config
        except Exception as e:
            print(f"Error loading config from {filename}: {e}")
            return {}

    @staticmethod
    def load_urls_from_txtfile(filename):
        with open(filename, 'r') as file:
            return [line.strip() for line in file.readlines()]

class ArgsHandler:
    def __init__(self):
        self.parser = self._get_parser()

    def _get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
        parser.add_argument("--urls", type=str, required=True, help="Path to the file containing URLs")


        parser.add_argument("--max_length", type=int, default=None)
        parser.add_argument("--n_dims", type=int, default=None)
        parser.add_argument("--n_heads", type=int, default=None)
        parser.add_argument("--n_layers", type=int, default=None)
        parser.add_argument("--intermediate_size", type=int, default=None)

        parser.add_argument("--name", type=str)
        parser.add_argument("--entity", type=str)
        parser.add_argument("--project", type=str)

        parser.add_argument("--gradient_accumulation_steps", type=int, default=256) # also used for block_size. max: 2,048
        parser.add_argument("--logging_steps", type=int, default=2) # also used for save_steps.
        parser.add_argument("--learning_rate", type=float, default=5e-4)
        parser.add_argument("--num_train_epochs", type=int, default=3)
        
        parser.add_argument("--auto_find_batch_size", default=True)
        parser.add_argument("--per_device_train_batch_size", type=int, default=128)
        parser.add_argument("--per_device_eval_batch_size", type=int, default=128)
        parser.add_argument("--do_eval", default=False)
        parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
        parser.add_argument("--fp16", default=True)
        parser.add_argument("--weight_decay", type=int, default=0.1)
        parser.add_argument("--save_total_limit", type=int, default=5)
        parser.add_argument("--overwrite_output_dir", default=True)
        parser.add_argument("--output_dir", type=str, default="./output") # checkpoints will be saved.
        parser.add_argument("--trained_dir", type=str, default="./trained") # trained model will be saved.
        parser.add_argument("--sample_size", type=int, default=-1)

        return parser
    
    def _update_args_with_config(self, args, config_from_file):
        args_dict = vars(args)
        for key, value in config_from_file.items():
            if key not in args_dict or args_dict[key] is None or args_dict[key] == self.parser.get_default(key):
                args_dict[key] = value
        return args_dict

    def _create_training_args(self, args_dict):
        training_args = TrainingArguments(
        gradient_accumulation_steps = args_dict['gradient_accumulation_steps'],
        logging_steps = args_dict['logging_steps'],
        save_steps = args_dict['logging_steps'],
        learning_rate = args_dict['learning_rate'],
        num_train_epochs = args_dict['num_train_epochs'],
        auto_find_batch_size = args_dict['auto_find_batch_size'],
        per_device_train_batch_size = args_dict['per_device_train_batch_size'],
        per_device_eval_batch_size = args_dict['per_device_eval_batch_size'],
        do_eval = args_dict['do_eval'],
        lr_scheduler_type = args_dict['lr_scheduler_type'],
        fp16 = args_dict['fp16'],
        weight_decay = args_dict['weight_decay'],
        save_total_limit = args_dict['save_total_limit'],
        overwrite_output_dir = args_dict['overwrite_output_dir'],
        output_dir = args_dict['output_dir']
        )
        return training_args

    def get_args(self):
        args = self.parser.parse_args()
        config_from_file = FileLoader.load_from_yamlfile(args.config)
        args_dict = self._update_args_with_config(args, config_from_file)
        training_args = self._create_training_args(args_dict)
        return args_dict, training_args

    
class ModelCreator:
    @staticmethod
    def create(model_class, args, tokenizer):
        model_kwargs = {
            'tokenizer': tokenizer,
            'max_length': args.get('max_length'),
            'n_dims': args.get('n_dims'),
            'n_heads': args.get('n_heads'),
            'n_layers': args.get('n_layers'),
            'intermediate_size': args.get('intermediate_size')
        }
        return model_class(**{k: v for k, v in model_kwargs.items() if v is not None})

if __name__ == "__main__":
    args_handler = ArgsHandler()
    args = args_handler.get_args()
    print(args)
