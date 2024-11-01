'''docstring to be adjusted later'''
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel

# 1. SentenceEncoder class with validation checks and positional encoding
class SentenceEncoder(nn.Module):
    '''Sentence Encoder with byte-level BPE tokenization, learned positional encoding, and validation checks'''
    def __init__(self, model, tokenizer, hidden_size):
        super(SentenceEncoder, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Adjust offset to zero if needed, or explicitly pass position_ids without offset if applicable
        if hasattr(self.model.decoder.embed_positions, 'offset'):
            self.model.decoder.embed_positions.offset = 0


    def forward(self, sentence):
        '''Encodes sentence into a sentence-level token with positional encoding'''
        # Tokenize sentence with truncation, respecting max token limit
        inputs = self.tokenizer(sentence, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=self.model.config.max_position_embeddings)
        
        # Validate tokenization output
        if "input_ids" not in inputs or inputs["input_ids"].size(1) == 0:
            raise ValueError("Tokenization failed or produced empty input_ids.")
        
        # Adjust position_ids to stay within the embedding matrix range without exceeding bounds
        max_pos_emb = min(self.model.config.max_position_embeddings, self.model.decoder.embed_positions.weight.shape[0]) - 1
        position_ids = torch.arange(0, inputs["input_ids"].shape[1]).clamp(max=max_pos_emb).unsqueeze(0).to(inputs["input_ids"].device)
        
        # Print position_ids right before embedding application for verification
        print("Final position IDs to be passed:", position_ids)
        print("Maximum embedding index:", max_pos_emb)

        # Pass tokens through the embedding layer of the OPT model
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract hidden states and apply feature fusion by summing them
        sentence_level_token = outputs.last_hidden_state.sum(dim=1)
        
        # Add positional embeddings
        positional_embeddings = self.model.decoder.embed_positions(position_ids)
        sentence_level_token = sentence_level_token + positional_embeddings
        
        # Apply layer normalization
        sentence_level_token = self.layer_norm(sentence_level_token)

        return sentence_level_token



# 2. SentenceDecoder class with validation checks
class SentenceDecoder(nn.Module):
    '''Sentence Decoder for generating sentences from sentence-level tokens, with validation checks'''
    def __init__(self, model, tokenizer, hidden_size):
        super(SentenceDecoder, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size

    def forward(self, sentence_level_token):
        '''Decodes sentence-level token back into a sentence'''
        decoded_sentence = []
        input_ids = self.tokenizer("<bos>", return_tensors="pt").input_ids.to(sentence_level_token.device)

        for _ in range(100):  # Maximum sentence length
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs.last_hidden_state
            next_token_id = torch.argmax(logits, dim=-1)[:, -1].item()
            
            # Break if end-of-sequence token is generated
            if next_token_id == self.tokenizer.eos_token_id:
                break
            
            # Validate next token ID
            if next_token_id >= self.tokenizer.vocab_size:
                raise ValueError("Next token ID is out of range.")
            
            decoded_sentence.append(next_token_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]]).to(input_ids.device)], dim=-1)

        return self.tokenizer.decode(decoded_sentence, skip_special_tokens=True)

# 3. EncoderDecoderModel class to encapsulate training logic with conditional parallelization
class EncoderDecoderModel:
    '''Encoder-Decoder Model with training logic, including focal loss and AdamW optimizer'''
    def __init__(self, encoder, decoder, tokenizer, learning_rate=1e-4):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=learning_rate
        )

        # Check for GPU availability and set up parallel training if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.encoder = nn.DataParallel(self.encoder).to(self.device)
            self.decoder = nn.DataParallel(self.decoder).to(self.device)
        else:
            self.encoder.to(self.device)
            self.decoder.to(self.device)

    def focal_loss(self, logits, targets, gamma=2.0):
        '''Calculates focal loss between logits and target tokens'''
        probas = torch.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=self.tokenizer.vocab_size).float().to(logits.device)
        loss = -targets_one_hot * ((1.0 - probas) ** gamma) * torch.log(probas)
        return loss.sum()

    def train(self, sentences):
        '''Training loop with focal loss and backpropagation'''
        for sentence in sentences:
            try:
                sentence_level_token = self.encoder(sentence).to(self.device)
                predicted_sentence = self.decoder(sentence_level_token)

                # Calculate logits from decoder output
                logits = predicted_sentence.logits  # Removed because predicted_sentence is a string
                targets = self.tokenizer(sentence, return_tensors="pt").input_ids[0].to(self.device)
                loss = self.focal_loss(logits, targets)  # Removed due to lack of logits

                # Perform backpropagation
                self.optimizer.zero_grad()
                loss.backward()  # Removed due to lack of loss calculation
                self.optimizer.step()
            except (ValueError, RuntimeError, TypeError) as e:
                print(f"Error during training on sentence '{sentence}': {e}")

# 4. InferenceEngine class for inference with sentence splitting
class InferenceEngine:
    '''Inference Engine for reconstructing text sentence-by-sentence'''
    def __init__(self, encoder, decoder, tokenizer):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer

    def inference(self, text):
        '''Performs inference on input text, reconstructing it sentence-by-sentence'''
        sentences = self.split_into_sentences(text)
        reconstructed_text = ""

        for sentence in sentences:
            sentence_level_token = self.encoder(sentence)
            decoded_sentence = self.decoder(sentence_level_token)
            reconstructed_text += decoded_sentence + " "

        return reconstructed_text.strip()

    def split_into_sentences(self, text):
        '''Splits text into sentences based on punctuation'''
        punctuation_marks = [".", "?", "!", ","]
        sentences = []
        current_sentence = ""

        for char in text:
            current_sentence += char
            if char in punctuation_marks:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        if current_sentence:
            sentences.append(current_sentence.strip())
        
        return sentences

# 5. Main function
def main():
    '''Main function to demonstrate end-to-end process with dynamic configuration'''
    hidden_size = 2048
    learning_rate = 1e-4
    MODEL_NAME = "facebook/opt-1.3b"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    
    # Print model configuration to verify max_position_embeddings
    print("Model configuration:", model.config)
    print("Max position embeddings:", model.config.max_position_embeddings)
    
    # Optionally, check the size of the positional embedding layer
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'embed_positions'):
        print("Positional embedding size:", model.decoder.embed_positions.weight.shape[0])
    else:
        print("Model does not have 'decoder.embed_positions'. Please verify model architecture.")

    # Initialize encoder and decoder
    encoder = SentenceEncoder(model, tokenizer, hidden_size)
    decoder = SentenceDecoder(model, tokenizer, hidden_size)
    
    # Initialize training model with optimizer and training configuration
    encoder_decoder_model = EncoderDecoderModel(encoder, decoder, tokenizer, learning_rate=learning_rate)
    
    # Explanation text used as data for training
    sentences = [
        "In this project, we are building a sentence-level variational autoencoder (SentenceVAE) combined with an OPT model for efficient next-sentence prediction.",
        "The encoder takes sentences, tokenizes them with byte-level BPE, and applies learned positional embeddings to produce sentence-level tokens.",
        "These tokens capture high-level sentence structure, which the model then uses in training and inference to reconstruct or predict sentences.",
        "We leverage masked self-attention and a focal loss function for improved training, and use parallel processing for enhanced performance if a GPU is available."
    ]

    # Train the model
    encoder_decoder_model.train(sentences)
    
    # Initialize inference engine
    inference_engine = InferenceEngine(encoder, decoder, tokenizer)
    
    # Perform inference on sample text
    text = "This is a test. How does it perform?"
    reconstructed_text = inference_engine.inference(text)
    
    # Output the result
    print("Reconstructed Text:", reconstructed_text)

# Entry point
if __name__ == "__main__":
    main()