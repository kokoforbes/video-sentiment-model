import torch
import torch.nn as nn
from transformers import BertModel  # type: ignore
from torchvision import models as vision_models  # type: ignore
from torchvision.models.video import R3D_18_Weights  # type: ignore

from meld_dataset import MELDDataset  # type: ignore


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        # Extract BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        pooler_output = outputs.pooler_output

        return self.projection(pooler_output)


class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = vision_models.video.r3d_18(
            weights=R3D_18_Weights.DEFAULT)

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_fts = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        # [batch_size, frames, channels, height, width] -> [batch_size, channels, frames, height, width]
        x = x.transpose(1, 2)
        return self.backbone(x)


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Lower level features
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Higher level features
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        for param in self.conv_layers.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = x.squeeze(1)  # Remove the channel dimension
        features = self.conv_layers(x)
        # features shape: [batch_size, 128, 1]
        # Remove the time dimension
        return self.projection(features.squeeze(-1))


class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize the encoders
        # Text, Video, and Audio encoders
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # fusion layer
        # Concatenate features from all modalities

        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Final classification layer
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            # 7 emotion classes - Anger, Disgust, Fear, Happiness, Sadness, Surprise, Neutral
            nn.Linear(64, 7)
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            # 3 sentiment classes - Negative, Neutral, Positive
            nn.Linear(64, 3)
        )

    def forward(self, text_inputs, video_frames, audio_features):

        text_features = self.text_encoder(
            text_inputs['input_ids'], text_inputs['attention_mask'])
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        # Concatenate features from all modalities
        combined_features = torch.cat(
            (text_features, video_features, audio_features), dim=1)
        # Pass through the fusion layer
        fused_features = self.fusion_layer(combined_features)
        # Pass through the classification layers
        emotion_output = self.emotion_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features)
        return {
            "emotions": emotion_output,
            "sentiments": sentiment_output
        }


class MultimodalTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Log dataset size
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        print("\n Dataset sizes:")
        print(f"Training samples: {train_size:,}")
        print(f"Validation samples: {val_size:,}")
        print(f"Batches per epoch: {len(train_loader):,}")

        # very high: 1, high: 0.1 - 0.01, medium: 1e-1, low: 1e-4, very low: 1e-5
        self.optimizer = torch.optim.Adam([
            {'params': model.text_encoder.parameters(), 'lr': 8e-6},
            {'params': model.video_encoder.parameters(), 'lr': 8e-5},
            {'params': model.audio_encoder.parameters(), 'lr': 8e-5},
            {'params': model.fusion_layer.parameters(), 'lr': 5e-4},
            {'params': model.emotion_classifier.parameters(), 'lr': 5e-4},
            {'params': model.sentiment_classifier.parameters(), 'lr': 5e-4}
        ], weight_decay=1e-5)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=2
        )

        self.emotion_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05
        )

        self.sentiment_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05
        )

        def train_epoch(self):
            self.model.train()
            running_loss = {'total': 0, 'emotion': 0, 'sentiment': 0}

            for batch in self.train_loader:
                device = next(self.model.parameters()).device
                text_inputs = {
                    'input_ids': batch['text_input']['input_ids'].to(device),
                    'attention_mask': batch['text_input']['attention_mask'].to(device)
                }
                video_frames = batch['video_frames'].to(device)
                audio_features = batch['audio_features'].to(device)
                emotion_labels = batch['emotion_labels'].to(device)
                sentiment_labels = batch['sentiment_labels'].to(device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(
                    text_inputs, video_frames, audio_features)
                # Compute the loss
                emotion_loss = self.emotion_criterion(
                    outputs['emotions'], emotion_labels)
                sentiment_loss = self.sentiment_criterion(
                    outputs['sentiments'], sentiment_labels)
                total_loss = emotion_loss + sentiment_loss

                # Backward pass
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                # Track losses
                running_loss['total'] += total_loss.item()
                running_loss['emotion'] += emotion_loss.item()
                running_loss['sentiment'] += sentiment_loss.item()

            return {K: v / len(self.train_loader) for K, v in running_loss.items()}

    def validate(self):
        self.model.eval()
        val_loss = {'total': 0, 'emotion': 0, 'sentiment': 0}
        all_emotion_preds = []
        all_emotion_labels = []
        all_sentiment_preds = []
        all_sentiment_labels = []

        with torch.inference_mode():
            for batch in self.val_loader:
                device = next(self.model.parameters()).device
                text_inputs = {
                    'input_ids': batch['text_input']['input_ids'].to(device),
                    'attention_mask': batch['text_input']['attention_mask'].to(device)
                }
                video_frames = batch['video_frames'].to(device)
                audio_features = batch['audio_features'].to(device)
                emotion_labels = batch['emotion_labels'].to(device)
                sentiment_labels = batch['sentiment_labels'].to(device)

                # Forward pass
                outputs = self.model(
                    text_inputs, video_frames, audio_features)

                # Compute the loss
                emotion_loss = self.emotion_criterion(
                    outputs['emotions'], emotion_labels)
                sentiment_loss = self.sentiment_criterion(
                    outputs['sentiments'], sentiment_labels)
                total_loss = emotion_loss + sentiment_loss

                all_emotion_preds.extend(
                    outputs['emotions'].argmax(dim=1).cpu().numpy())
                all_emotion_labels.extend(
                    emotion_labels.cpu().numpy())
                all_sentiment_preds.extend(
                    outputs['sentiments'].argmax(dim=1).cpu().numpy())
                all_sentiment_labels.extend(
                    sentiment_labels.cpu().numpy())

                # Track losses
                val_loss['total'] += total_loss.item()
                val_loss['emotion'] += emotion_loss.item()
                val_loss['sentiment'] += sentiment_loss.item()


if __name__ == "__main__":
    dataset = MELDDataset(
        '../dataset/train/train_sent_emo.csv', '../dataset/train/train_splits')

    sample = dataset[0]
    model = MultimodalSentimentModel()
    model.eval()

    text_inputs = {
        'input_ids': sample['text_inputs']['input_ids'].unsqueeze(0),
        'attention_mask': sample['text_inputs']['attention_mask'].unsqueeze(0)
    }

    video_frames = sample['video_frames'].unsqueeze(0)
    audio_features = sample['audio_features'].unsqueeze(0)

    with torch.inference_mode():
        outputs = model(text_inputs, video_frames, audio_features)

        emotion_probs = torch.softmax(outputs['emotions'], dim=1)[0]
        sentiment_probs = torch.softmax(outputs['sentiments'], dim=1)[0]

    emotion_map = {
        0: 'anger',
        1: 'disgust',
        2: 'fear',
        3: 'joy',
        4: 'neutral',
        5: 'sadness',
        6: 'surprise'
    }

    sentiment_map = {
        0: 'negative',
        1: 'neutral',
        2: 'positive'
    }

    # Print the predicted emotions and sentiments
    for i, prob in enumerate(emotion_probs):
        print(f"{emotion_map[i]}: {prob.item():.2f}")
    for i, prob in enumerate(sentiment_probs):
        print(f"{sentiment_map[i]}: {prob.item():.2f}")

    # print("Predictions for utterance:")
