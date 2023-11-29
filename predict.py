"""
download model weights to /data
wget wget -O - https://pyannote-speaker-diarization.s3.eu-west-2.amazonaws.com/data-2023-03-25-02.tar.gz | tar xz -C /
"""

from cog import BasePredictor, Input, Path, BaseModel
from pyannote.audio.pipelines import SpeakerDiarization

from lib.diarization import DiarizationPostProcessor
from lib.audio import AudioPreProcessor
import torch


class SpeakerSegment(BaseModel):
    speaker: str
    start: str
    stop: str


class Speakers(BaseModel):
    count: int
    labels: list[str]
    embeddings: dict[str, list[float]]


class ModelOutput(BaseModel):
    segments: list[SpeakerSegment]
    speakers: Speakers


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.diarization = SpeakerDiarization(
            segmentation="/pyannote/segmentation-3.0/pytorch_model.bin",
            embedding="/hbredin/wespeaker-voxceleb-resnet34-LM/speaker-embedding.onnx",
            clustering="AgglomerativeClustering",
            segmentation_batch_size=32,
            embedding_batch_size=1,
            embedding_exclude_overlap=True,
        )
        self.diarization.to(torch.device("cuda"))
        self.diarization.instantiate(
            {
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": 12,
                    "threshold": 0.7045654963945799,
                },
                "segmentation": {
                    "min_duration_off": 0.0,
                },
            }
        )
        self.diarization_post = DiarizationPostProcessor()
        self.audio_pre = AudioPreProcessor()

    def run_diarization(self):
        closure = {"embeddings": None}

        def hook(name, *args, **kwargs):
            if name == "embeddings" and len(args) > 0:
                closure["embeddings"] = args[0]

        print("diarizing audio file...")
        diarization = self.diarization(self.audio_pre.output_path, hook=hook)
        chunk_duration = self.diarization._segmentation.model.specifications.duration
        embeddings = {
            "data": closure["embeddings"],
            "chunk_duration": chunk_duration,
            "chunk_offset": self.diarization.segmentation_step * chunk_duration,
        }
        return self.diarization_post.process(diarization, embeddings)

    def predict(
        self,
        audio: Path = Input(
            description="Audio file or url",
            default="https://replicate.delivery/pbxt/IZjTvet2ZGiyiYaMEEPrzn0xY1UDNsh0NfcO9qeTlpwCo7ig/lex-levin-4min.mp3",
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""

        self.audio_pre.process(audio)

        if self.audio_pre.error:
            print(self.audio_pre.error)
            result = self.diarization_post.empty_result()
        else:
            result = self.run_diarization()

        self.audio_pre.cleanup()

        return ModelOutput(**result)
