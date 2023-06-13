from arguments import InitializationArguments
from transformers import DeiTConfig, RobertaConfig, VisionEncoderDecoderConfig,VisionEncoderDecoderModel, TrOCRProcessor, AutoTokenizer, HfArgumentParser


# Configuration
parser = HfArgumentParser(InitializationArguments)
args = parser.parse_args()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

# Load processor
processor = TrOCRProcessor.from_pretrained(args.processor_name)
processor.tokenizer = tokenizer

config_encoder = DeiTConfig()
config_decoder = RobertaConfig.from_pretrained(args.decoder_name)
# config_decoder = RobertaConfig()


config_encoder.image_size = 384

config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)


config.decoder_start_token_id = processor.tokenizer.cls_token_id
config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
config.vocab_size = config.decoder.vocab_size

# set beam search parameters
config.eos_token_id = processor.tokenizer.sep_token_id
# config.max_length = args.max_length
config.max_new_tokens = args.max_length
config.early_stopping = True
config.no_repeat_ngram_size = 3
config.length_penalty = 2.0
config.num_beams = 4

# Load model
model = VisionEncoderDecoderModel(config=config)

# Save model to the hub
model.save_pretrained(args.model_name, push_to_hub=args.push_to_hub)
processor.save_pretrained(args.model_name, push_to_hub=args.push_to_hub)


