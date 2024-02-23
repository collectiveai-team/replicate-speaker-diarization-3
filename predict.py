"""
download model weights to /data
wget wget -O - https://pyannote-speaker-diarization.s3.eu-west-2.amazonaws.com/data-2023-03-25-02.tar.gz | tar xz -C /
"""

import torch
import torchaudio
from cog import Path, Input, BaseModel, BasePredictor
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.pipelines.utils.hook import ProgressHook

from lib.audio import AudioPreProcessor
from lib.diarization import DiarizationPostProcessor


class SpeakerSegment(BaseModel):
    speaker: str
    start: str
    stop: str


class Speakers(BaseModel):
    count: int
    labels: list[str]
    embeddings: dict[str, list[float]]


class Output(BaseModel):
    segments: list[SpeakerSegment]
    speakers: Speakers


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("running in gpu")
        else:
            self.device = torch.device("cpu")
            print("running in cpu")

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

        print("starting diarizing...")
        print("> loading audio file")
        waveform, sample_rate = torchaudio.load(self.audio_pre.output_path)

        print("> diarizing audio file")
        with ProgressHook() as progress_hook:
            closure = {"embeddings": None}

            def hook(name, *args, **kwargs):
                if name == "embeddings" and len(args) > 0:
                    closure["embeddings"] = args[0]
                progress_hook(name, *args, **kwargs)

            diarization = self.diarization(
                {"waveform": waveform, "sample_rate": sample_rate},
                hook=hook,
            )
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
    ) -> Output:
        """Run a single prediction on the model"""

        print(">> received audio file:", audio)
        self.audio_pre.process(audio)

        if self.audio_pre.error:
            print(self.audio_pre.error)
            result = self.diarization_post.empty_result()
        else:
            result = self.run_diarization()

        self.audio_pre.cleanup()

        return Output(**result)
