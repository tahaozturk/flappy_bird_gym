import os
import json
from moviepy import VideoFileClip, TextClip, ColorClip, CompositeVideoClip, concatenate_videoclips

def process_videos():
    base_dir = "."
    videos_dir = os.path.join(base_dir, "videos")
    output_dir = os.path.join(base_dir, "edited_videos")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    categories = {
        "small": "Small",
        "medium": "Mid",
        "large": "Big"
    }

    font_path = '/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf'

    for folder_name, display_name in categories.items():
        folder_path = os.path.join(videos_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"Directory {folder_path} does not exist, skipping.")
            continue

        print(f"Processing category: {display_name} ({folder_name})")
        clips_data = []

        # Gather video and metadata pairs
        for filename in os.listdir(folder_path):
            if filename.endswith(".mp4"):
                base_name = filename[:-4]
                meta_file = base_name + ".meta.json"
                meta_path = os.path.join(folder_path, meta_file)
                video_path = os.path.join(folder_path, filename)

                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                        episode_id = meta.get("episode_id", 0)
                        
                        # Calculate global training step based on episode index and eval frequency
                        eval_freq = {
                            "small": 12500,
                            "medium": 25000,
                            "large": 50000
                        }[folder_name]
                        # Each episode corresponds to 1 evaluation step
                        actual_step = (episode_id + 1) * eval_freq
                        
                        clips_data.append({
                            "video_path": video_path,
                            "step_id": actual_step,
                            "episode_id": episode_id
                        })
                else:
                    print(f"Warning: Metadata not found for {filename}")

        if not clips_data:
            print(f"No valid clips found in {folder_name}")
            continue

        # Sort chronologically (by step_id, then episode_id)
        clips_data.sort(key=lambda x: (x["step_id"], x["episode_id"]))

        processed_clips = []
        for data in clips_data:
            try:
                # Load video
                clip = VideoFileClip(data["video_path"])
                
                # Strip audio and Speed up 2x
                clip = clip.without_audio()
                fast_clip = clip.with_speed_scaled(2.0)

                # Create text overlay
                text = f"Model: {display_name}\nStep: {data['step_id']}"
                txt_clip = TextClip(text=text, font=font_path, font_size=18, color='white', text_align='left', margin=(5, 5))
                
                # Semi-transparent background for text
                bg_clip = ColorClip(size=(txt_clip.size[0] + 10, txt_clip.size[1] + 10), color=(0, 0, 0))
                bg_clip = bg_clip.with_opacity(0.6)

                # Composite background and text
                txt_comp = CompositeVideoClip([bg_clip, txt_clip.with_position('center')])
                
                # Set duration and position on main video (bottom-left, above edge)
                y_pos = fast_clip.size[1] - txt_comp.size[1] - 10
                txt_comp = txt_comp.with_duration(fast_clip.duration).with_position((10, y_pos))

                # Final clip with overlay
                final_clip = CompositeVideoClip([fast_clip, txt_comp])
                processed_clips.append(final_clip)
            except Exception as e:
                print(f"Error processing {data['video_path']}: {e}")

        if processed_clips:
            print(f"Concatenating {len(processed_clips)} clips for {display_name}...")
            # concatenate_videoclips with method="compose" helps if clips differ slightly
            final_master = concatenate_videoclips(processed_clips, method="compose")
            
            output_file = os.path.join(output_dir, f"master_{folder_name}.mp4")
            print(f"Saving to {output_file}...")
            final_master.write_videofile(output_file, codec="libx264", fps=30, preset="fast", threads=4)
            
            # Close clips to free memory
            for c in processed_clips:
                c.close()
            final_master.close()

if __name__ == "__main__":
    process_videos()
