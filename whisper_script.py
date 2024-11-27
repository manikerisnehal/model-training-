import whisper
import warnings
import spacy
from spacy.training import Example
from spacy.util import minibatch
import random
import json
from deepmultilingualpunctuation import PunctuationModel

warnings.filterwarnings("ignore")

def transcribe_audio(file_path):
    """
    Transcribe audio to text using the Whisper model.
    :param file_path: Path to the audio file.
    :return: Raw transcription text.
    """
    try:
        model = whisper.load_model("turbo")  # Use the desired Whisper model
        result = model.transcribe(file_path, language="en")
        print("\nRaw Transcription:\n", result["text"])
        return result["text"]
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def restore_punctuation(text):
    """
    Restore punctuation in the transcribed text using a pre-trained model.
    :param text: Raw transcribed text without punctuation.
    :return: Text with restored punctuation.
    """
    try:
        model = PunctuationModel()
        punctuated_text = model.restore_punctuation(text)
        print("\nPunctuated Text:\n", punctuated_text)
        return punctuated_text
    except Exception as e:
        print(f"Error during punctuation restoration: {e}")
        return text

def train_spacy_model(training_data_path, output_model_path):
    """
    Train a fine-tuned spaCy NER model using the provided training data.
    :param training_data_path: Path to the training data JSON file.
    :param output_model_path: Directory to save the fine-tuned model.
    """
    nlp = spacy.load("en_core_web_sm")
    ner = nlp.get_pipe("ner")

    with open(training_data_path, "r") as file:
        data = json.load(file)

    # Add new labels
    for item in data:
        for ent in item["entities"]:
            ner.add_label(ent[2])

    # Prepare training data
    training_data = []
    for item in data:
        doc = nlp.make_doc(item["text"])
        entities = {"entities": item["entities"]}
        example = Example.from_dict(doc, entities)
        training_data.append(example)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()
        for iteration in range(100):
            random.shuffle(training_data)
            losses = {}
            batches = minibatch(training_data, size=4)
            for batch in batches:
                nlp.update(batch, drop=0.3, losses=losses)
            print(f"Iteration {iteration + 1}, Loss: {losses['ner']}")

    nlp.to_disk(output_model_path)
    print(f"Fine-tuned model saved to '{output_model_path}'")

def test_fine_tuned_model(model_path, text):
    """
    Test the fine-tuned spaCy model on the given text.
    :param model_path: Path to the fine-tuned model.
    :param text: Input text for entity recognition.
    """
    nlp = spacy.load(model_path)
    doc = nlp(text)
    print("\nExtracted Entities:")
    for ent in doc.ents:
        print(f"{ent.label_}: {ent.text}")

def main():
    audio_file = r"E:\whisper\IPDRound.ogg"  # Step 1: Specify the audio file path

    # Step 2: Transcribe audio to text
    raw_transcription = transcribe_audio(audio_file)

    if raw_transcription:
        # Step 3: Restore punctuation
        punctuated_text = restore_punctuation(raw_transcription)

        # Step 4: Train spaCy model (Ensure training data exists)
        training_data_path = "training_data.json"
        output_model_path = "fine_tuned_clinical_model"
        train_spacy_model(training_data_path, output_model_path)

        # Step 5: Test fine-tuned model using the punctuated text
        test_fine_tuned_model(output_model_path, punctuated_text)

if __name__ == "__main__":
    main()
