import whisper
from pyannote.audio import Pipeline
import torch
import os
import argparse
from collections import defaultdict
import re

hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

def transcribe_audio(audio_path, model_name="base"):
    """Transcribes the audio file using Whisper."""
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    print("Transcribing audio...")
    # Need word timestamps for speaker assignment
    result = model.transcribe(audio_path, word_timestamps=True)
    print("Transcription complete.")
    return result

def diarize_audio(audio_path, hf_token=hf_token):
    """Performs speaker diarization using pyannote.audio."""
    if hf_token is None:
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if hf_token is None:
        raise ValueError("Hugging Face Hub token not found. "
                         "Please set the HUGGING_FACE_HUB_TOKEN environment variable "
                         "or pass it as an argument.")

    print("Loading diarization pipeline...")
    # Use pyannote/speaker-diarization-3.1 for potentially better accuracy
    # pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)

    # Using older model for potentially less stringent requirements if issues arise
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)

    # Send pipeline to GPU if available
    if torch.cuda.is_available():
        print("Moving diarization pipeline to GPU...")
        pipeline = pipeline.to(torch.device("cuda"))

    print("Performing speaker diarization...")
    diarization = pipeline(audio_path)
    print("Diarization complete.")
    return diarization

def assign_speakers_to_words(transcription_result, diarization_result):
    """Assigns speakers to transcribed words based on timestamps."""
    print("Assigning speakers to words...")
    word_segments = []
    for segment in transcription_result["segments"]:
        for word_info in segment.get("words", []):
            # word_info format: {'word': 'Hello', 'start': 0.5, 'end': 0.8}
            word_segments.append(word_info)

    speaker_assignments = []
    diarization_turns = list(diarization_result.itertracks(yield_label=True))

    word_idx = 0
    turn_idx = 0

    while word_idx < len(word_segments) and turn_idx < len(diarization_turns):
        turn_segment, _, speaker = diarization_turns[turn_idx]
        word_info = word_segments[word_idx]

        word_start = word_info['start']
        word_end = word_info['end']
        turn_start = turn_segment.start
        turn_end = turn_segment.end

        # Check if the word midpoint falls within the current speaker turn
        word_midpoint = (word_start + word_end) / 2

        if turn_start <= word_midpoint < turn_end:
            speaker_assignments.append({
                'word': word_info['word'],
                'start': word_start,
                'end': word_end,
                'speaker': speaker
            })
            word_idx += 1
        elif word_midpoint >= turn_end:
            # Move to the next speaker turn
            turn_idx += 1
        else: # word_midpoint < turn_start
             # This word doesn't seem to belong to the current turn or any previous turn
             # Assign as 'UNKNOWN' or handle as needed (e.g., assign to nearest turn)
             # For simplicity, we'll skip for now or mark as unknown
             speaker_assignments.append({
                 'word': word_info['word'],
                 'start': word_start,
                 'end': word_end,
                 'speaker': 'UNKNOWN' # Or handle differently
             })
             word_idx += 1

    # Handle any remaining words that might not have been assigned
    while word_idx < len(word_segments):
         word_info = word_segments[word_idx]
         speaker_assignments.append({
            'word': word_info['word'],
            'start': word_info['start'],
            'end': word_info['end'],
            'speaker': 'UNKNOWN'
         })
         word_idx += 1

    print("Speaker assignment complete.")
    return speaker_assignments

def detect_confusion(text):
    """Detects potential confusion in a text segment using enhanced heuristics."""
    text_lower = text.lower().strip()
    if not text_lower:
        return False

    # More comprehensive lists
    confusion_keywords = [
        "huh", "what?", "confused", "don't understand", "didn't get that",
        "sorry?", "pardon?", "i'm lost", "not following", "didn't catch that",
        "say again?", "what was that?", "i don't get it", "not sure i understand",
        "what does that mean", "could you repeat"
    ]
    # Added common filler phrases and words often associated with hesitation/uncertainty
    filler_words = ["uh", "um", "uhh", "umm", "err", "like", "you know", "actually", "basically", "sort of", "kind of"]
    question_starters = ["how", "why", "what", "when", "who", "where", "is", "are", "do", "does", "did", "can", "could", "would", "should"]

    score = 0

    # 1. Check for direct confusion keywords/phrases (high weight)
    for keyword in confusion_keywords:
        if keyword in text_lower:
            score += 3 # Higher score for direct confusion indicators

    # 2. Check for excessive filler words/phrases (moderate weight)
    filler_count = sum(text_lower.count(fw) for fw in filler_words)
    # Normalize by text length to avoid penalizing longer segments unfairly
    words = re.findall(r'\w+', text_lower)
    num_words = len(words)
    if num_words > 0 and (filler_count / num_words) > 0.15: # If fillers are > 15% of words
        score += 1
    elif filler_count > 3: # Or if absolute count is high in shorter segments
        score += 1


    # 3. Check for questions (moderate weight)
    # Check end punctuation or if sentence starts with a question word
    starts_with_question_word = any(text_lower.startswith(qw + " ") for qw in question_starters)
    if text_lower.endswith('?') or starts_with_question_word:
        score += 1
        # Bonus if it's a short question (often indicates quick confusion/clarification)
        if num_words < 5:
            score +=1


    # 4. Check for repetition (lower weight) - Check for repeated trigrams
    if num_words > 5:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        # Count occurrences of each trigram
        trigram_counts = defaultdict(int)
        for tg in trigrams:
            trigram_counts[tg] += 1
        # If any trigram appears more than once
        if any(count > 1 for count in trigram_counts.values()):
             score += 1 # Penalize repetition slightly

    # 5. Check for very short utterances (often indicate quick interjection like "Huh?")
    if num_words <= 2 and not text_lower.endswith('.'): # Avoid penalizing short statements like "Yes."
        # Check if it contains a filler or confusion word, or is just a question word
        if filler_count > 0 or score > 0 or text_lower in question_starters:
             score += 1


    # Adjust threshold based on refinements
    # print(f"DEBUG: Text='{text}', Score={score}") # Optional: for debugging
    return score >= 3 # Return True if score reaches the threshold

def format_output(speaker_word_assignments):
    """Formats the speaker-assigned words into a readable transcript with confusion detection."""
    print("\n--- Transcript with Speaker Diarization and Confusion Detection ---")
    current_speaker = None
    current_segment_text = []
    current_segment_start = 0

    for i, item in enumerate(speaker_word_assignments):
        speaker = item['speaker']
        word = item['word']
        start = item['start']

        if current_speaker is None:
            current_speaker = speaker
            current_segment_start = start

        if speaker != current_speaker:
            # End of the previous segment
            segment_text = " ".join(current_segment_text).strip()
            is_confused = detect_confusion(segment_text)
            confusion_marker = " [CONFUSED?]" if is_confused else ""
            print(f"{current_speaker} ({current_segment_start:.2f}s): {segment_text}{confusion_marker}")

            # Start of a new segment
            current_speaker = speaker
            current_segment_text = [word]
            current_segment_start = start
        else:
            current_segment_text.append(word)

        # Print the last segment
        if i == len(speaker_word_assignments) - 1:
            segment_text = " ".join(current_segment_text).strip()
            is_confused = detect_confusion(segment_text)
            confusion_marker = " [CONFUSED?]" if is_confused else ""
            print(f"{current_speaker} ({current_segment_start:.2f}s): {segment_text}{confusion_marker}")

    print("--- End of Transcript ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio, perform speaker diarization, and detect confusion.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file (e.g., wav, mp3).")
    parser.add_argument("--whisper-model", type=str, default="base", help="Name of the Whisper model to use (e.g., tiny, base, small, medium, large). Default: base")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face Hub token for pyannote.audio. Defaults to HUGGING_FACE_HUB_TOKEN env var.")
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found at {args.audio_file}")
        exit(1)

    # 1. Transcribe
    transcription = transcribe_audio(args.audio_file, model_name=args.whisper_model)

    # 2. Diarize
    try:
        diarization = diarize_audio(args.audio_file, hf_token=args.hf_token)
    except Exception as e:
        print(f"Error during diarization: {e}")
        print("Proceeding with transcription only.")
        print("\n--- Full Transcription (No Diarization) ---")
        print(transcription["text"])
        exit(1)

    # 3. Assign Speakers
    speaker_word_assignments = assign_speakers_to_words(transcription, diarization)

    # 4. Format Output with Confusion Detection
    format_output(speaker_word_assignments) 