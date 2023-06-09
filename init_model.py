from arguments import InitializationArguments
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AutoTokenizer, HfArgumentParser


# Configuration
parser = HfArgumentParser(InitializationArguments)
args = parser.parse_args()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

# Load processor
processor = TrOCRProcessor.from_pretrained(args.processor_name)
processor.tokenizer = tokenizer

# Load model
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(args.encoder_name, args.decoder_name)

# Save model to the hub
model.save_pretrained(args.model_name, push_to_hub=args.push_to_hub)
processor.save_pretrained(args.model_name, push_to_hub=args.push_to_hub)