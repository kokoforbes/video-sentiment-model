from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer  # type: ignore
import os
class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir ):
        self.data = pd.read_csv(csv_path)
       
        self.video_dir = video_dir
    
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear
        self.emotion_map = {
            "anger": 0,
            "disgust": 1,
            "sadness": 2,
            "joy": 3,
            "neutral": 4,
            "surprise": 5,
            "fear": 6
        }

        self.sentiment_map = {
            "negative": 0,
             "neutral": 1,
            "positive": 2,
           
        }

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_filename = f"""dia{row['dialogue_id']}_utt{
            row['utterance_id']}.mp4"""
        path = os.path.join(self.video_dir, video_filename)
        video_path = os.path.exists(path)

        if not video_path:
            raise FileNotFoundError(f"Video file not found for filename: {path}")
            
        print("file found")

if __name__ == "__main__":
    meld = MELDDataset('../dataset/dev/dev_sent_emo.csv', '../dataset/dev/dev_splits_complete')
  
    print(meld[0])