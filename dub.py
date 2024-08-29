import os
import scipy.io.wavfile
from spleeter.separator import Separator
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, vfx
import whisperx
from tqdm import tqdm
import torch
from pydub import AudioSegment
from bark import SAMPLE_RATE, generate_audio, preload_models

# Function to process the video and generate the output with synchronized audio
def process_video(video_path):
    # Extract base name (without extension) from video file path
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Define dynamic directories based on the base name
    split_dir = os.path.join("split", base_name)
    output_dir = os.path.join("translated_audio_segments", base_name)
    
    # Initialize Separator
    separator = Separator('spleeter:2stems')
    separator.separate_to_file(video_path, split_dir)

    # Load and transcribe audio using WhisperX
    device = "cuda"
    batch_size = 4 
    compute_type = "float16"
    audio_path = os.path.join(split_dir, "vocals.wav")
    audio = whisperx.load_audio(audio_path)
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    result = model.transcribe(audio, batch_size=batch_size)

    # Align and diarize
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    diarize_model = whisperx.DiarizationPipeline(use_auth_token="", device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Process segments and prepare translations
    segments_list = []
    accumulated_content = ""
    max_duration = 10.0

    for i in range(len(result['segments'])):
        current_segment = result['segments'][i]
        previous_segment_text = result['segments'][i-1]['text'] if i > 0 else ''
        accumulated_content += " " + current_segment['text']
        next_segment_text = result['segments'][i+1]['text'] if i < len(result['segments']) - 1 else ''
        duration = current_segment['end'] - current_segment['start']
        speaker = current_segment.get('speaker', 'Unknown')

        if duration > max_duration:
            midpoint = len(current_segment['text']) // 2
            part1_text = current_segment['text'][:midpoint].strip()
            part2_text = current_segment['text'][midpoint:].strip()

            segment_info_part1 = {
                'text': part1_text,
                'start_time': current_segment['start'],
                'end_time': current_segment['start'] + duration / 2,
                'duration': duration / 2,
                'accumulated_content': accumulated_content.strip(),
                'next_text': part2_text,
                'speaker': speaker,
            }
            segments_list.append(segment_info_part1)

            segment_info_part2 = {
                'text': part2_text,
                'start_time': current_segment['start'] + duration / 2,
                'end_time': current_segment['end'],
                'duration': duration / 2,
                'accumulated_content': accumulated_content.strip(),
                'next_text': next_segment_text,
                'speaker': speaker,
            }
            segments_list.append(segment_info_part2)
        else:
            segment_info = {
                'text': current_segment['text'],
                'start_time': current_segment['start'],
                'end_time': current_segment['end'],
                'duration': duration,
                'accumulated_content': accumulated_content.strip(),
                'next_text': next_segment_text,
                'speaker': speaker,
            }
            segments_list.append(segment_info)

    # Translate segments using OpenAI GPT
    openai.api_key = ""  # Add your OpenAI API key here
    translations = []
    for segment in segments_list:
        prompt = f"""
        Translate the following English segment into Hindi while preserving the context, maintaining the narrative flow, staying within the given duration, and keeping proper names in their original form:
        {segment['text']}
        """
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        translated_text = chat_completion.choices[0].message['content'].strip()
        translation = {
            'text': translated_text,
            'start_time': segment['start_time'],
            'end_time': segment['end_time'],
            'speaker': segment['speaker']
        }
        translations.append(translation)

    # Generate audio for translations using Bark
    preload_models()
    unique_speakers = diarize_segments.speaker.unique()
    speaker_mapping = {}
    for i, speaker in enumerate(unique_speakers):
        speaker_mapping[speaker] = f"v2/hi_speaker_{i}"

    os.makedirs(output_dir, exist_ok=True)

    for translation in translations:
        start_time_str = f"{translation['start_time']:.2f}"
        end_time_str = f"{translation['end_time']:.2f}"
        filename = f"{start_time_str}-{end_time_str}.wav"
        output_path = os.path.join(output_dir, filename)
        speaker_label = translation.get('speaker', None)
        history_prompt = speaker_mapping.get(speaker_label, "v2/hi_speaker_1")
        tr_audio = generate_audio(translation['text'], history_prompt=history_prompt)
        scipy.io.wavfile.write(output_path, rate=SAMPLE_RATE, data=tr_audio)

    # Merge the new audio segments with the original video
    video = VideoFileClip(video_path)
    video_no_audio = video.without_audio()
    background_audio = AudioFileClip(os.path.join(split_dir, "accompaniment.wav"))
    video_with_new_audio = video_no_audio.set_audio(background_audio)
    video_with_new_audio.write_videofile(f"{base_name}_video_with_new_audio.mp4")

    audio_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
    audio_files.sort(key=lambda x: float(x.split('-')[0]))

    video_segments = []
    previous_end_time = 0.0

    for audio_file in audio_files:
        start_time, end_time = map(float, audio_file[:-4].split('-'))
        if start_time > previous_end_time:
            missing_segment = video.subclip(previous_end_time, start_time)
            video_segments.append(missing_segment)
        
        audio = AudioFileClip(os.path.join(output_dir, audio_file))
        target_duration = audio.duration
        video_segment = video.subclip(start_time, end_time)
        current_duration = video_segment.duration
        speed_factor = current_duration / target_duration
        modified_segment = video_segment.fx(vfx.speedx, speed_factor)
        modified_segment = modified_segment.set_audio(audio)
        video_segments.append(modified_segment)
        previous_end_time = end_time

    if previous_end_time < video.duration:
        final_segment = video.subclip(previous_end_time, video.duration)
        video_segments.append(final_segment)

    final_video = concatenate_videoclips(video_segments)
    final_video.write_videofile(f'{base_name}_final_output_video.mp4', codec="libx264")

if __name__ == "__main__":
    video_path = "path/to/your/video.mp4"  # Replace with your video path
    process_video(video_path)
