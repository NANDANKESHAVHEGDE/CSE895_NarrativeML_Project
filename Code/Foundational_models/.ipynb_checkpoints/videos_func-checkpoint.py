import os
import cv2
from tqdm import tqdm

def read_video(video_path):
    """
    Reads a video from the specified path.

    Args:
        video_path (str): Path to the mp4 video file.

    Returns:
        cv2.VideoCapture: OpenCV video capture object.
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    return video

def get_video_length(video):
    """
    Determines the length of the video in seconds.

    Args:
        video (cv2.VideoCapture): OpenCV video capture object.

    Returns:
        float: Length of the video in seconds.
    """
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    return frame_count / fps

def cut_video_into_segments(video, num_segments):
    """
    Splits the video into equal segments.

    Args:
        video (cv2.VideoCapture): OpenCV video capture object.
        num_segments (int): Number of segments to divide the video into.

    Returns:
        list of tuple: List containing start and end frames for each segment.
    """
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    segment_length = total_frames // num_segments

    segments = []
    for i in range(num_segments):
        start_frame = i * segment_length
        end_frame = start_frame + segment_length
        if i == num_segments - 1:
            end_frame = total_frames  # Include any leftover frames in the last segment
        segments.append((start_frame, end_frame))
    return segments

def extract_frames(video, start_frame, end_frame, num_frames):
    """
    Extracts consecutive frames from a video segment.

    Args:
        video (cv2.VideoCapture): OpenCV video capture object.
        start_frame (int): Starting frame number of the segment.
        end_frame (int): Ending frame number of the segment.
        num_frames (int): Number of consecutive frames to extract.

    Returns:
        list of numpy.ndarray: List of extracted frames.
    """
    step = (end_frame - start_frame) // num_frames
    frames = []
    for i in range(num_frames):
        frame_number = start_frame + i * step
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video.read()
        if ret:
            frames.append(frame)
    return frames

def save_frames(frames, output_dir):
    """
    Saves frames as individual image files.

    Args:
        frames (list of numpy.ndarray): List of frames to save.
        output_dir (str): Directory to save the frames.

    """
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        file_path = os.path.join(output_dir, f"frame_{i+1}.jpg")
        cv2.imwrite(file_path, frame)

def process_video(video_path, output_dir, num_segments=8, frames_per_segment=16, short:int=None, long:int=None):
    """
    Main function to process the video and save frames.

    Args:
        video_path (str): Path to the mp4 video file.
        output_dir (str): Directory to save the frames.
        num_segments (int): Number of segments to divide the video into.
        short (int): if the video are shorter than this number (in second) then it is classified as short video.
        long (int): if the video are longer or equal to this number (in second) then it is classified as long video.
        frames_per_segment (int): Number of frames to extract from each segment.
    """
    video = read_video(video_path)
    #Redefine num_segment if short and long are defined
    if short is not None and long is not None:
        #Compute the video duration
        num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = video.get(cv2.CAP_PROP_FPS)
        seconds = num_frames / fps
        #Redefine the num_segments and frames_per_segment
        if seconds < short:
            num_segments_ = (num_segments // 2) // 2
            frames_per_segment_ = (frames_per_segment // 2) - 2
        elif short <= seconds and seconds < long:
            num_segments_ = num_segments // 2
            frames_per_segment_ = frames_per_segment // 2
        else:
            num_segments_ = num_segments
            frames_per_segment_ = frames_per_segment 

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    print(num_segments_)
    segments = cut_video_into_segments(video, num_segments_)

    for i, (start_frame, end_frame) in enumerate(segments):
        segment_output_dir = os.path.join(output_dir, f"{video_id}_segment_{i+1}")
        frames = extract_frames(video, start_frame, end_frame, frames_per_segment_)
        save_frames(frames, segment_output_dir)

    video.release()

def videos_framing(videos_dir, frames_dir=None, video_extension='mp4', num_segments:int=8, frames_per_segment:int=16, short:int=None,
                   long:int=None):
    video_ids = os.listdir(videos_dir)
    video_ids = sorted(video_ids)
    video_ids = video_ids[1:]
    for video_id in tqdm(video_ids):
        video_path = os.path.join(videos_dir, video_id, video_id + "." + video_extension)
        frame_dir = os.path.join(frames_dir, video_id)
        os.makedirs(frame_dir, exist_ok=True)
        process_video(video_path, frame_dir, num_segments, frames_per_segment, short=short, long=long)
    print("Done procesing all videos into frames!")