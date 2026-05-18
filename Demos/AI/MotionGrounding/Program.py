# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path

import torch
from ai4animation import (
    Actor,
    AI4Animation,
    DataSampler,
    Dataset,
    FeedTensor,
    MirrorModule,
    MotionEditor,
    MotionModule,
    MultiLayerPerceptron,
    Plotting,
    ReadTensor,
    RootModule,
    Tensor,
    TimeSeries,
    Transform,
    Utility,
    Vector3,
)

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Cranberry")

sys.path.append(ASSETS_PATH)
import Definitions

# Reduced epoch count from 150 to 100 for faster experimentation on my machine
EPOCH_COUNT = 100
# Reduced batch size from 32 to 16 to avoid OOM on my GPU (8GB VRAM)
BATCH_SIZE = 16
FRAMERATE = 30
DRAW_INTERVAL = 500
BONES = Definitions.FULL_BODY_NAMES
SMOOTHING_WINDOW = 2.0
RESOLUTION = 11
INPUT_DIM = RESOLUTION * len(BONES) * 9
OUTPUT_DIM = len(Definitions.FULL_BODY_NAMES) * 9

MAX_FILES = None


class Program:
    def Start(self):
        Utility.SetSeed(23456)

        self.Dataset = Dataset(
            os.path.join(ASSETS_PATH, "Motions"),
            [
                lambda x: RootModule(
                    x,
                    Definitions.HipName,
                    Definitions.LeftHipName,
                    Definitions.RightHipName,
                    Definitions.LeftShoulderName,
                    Definitions.RightShoulderName,
                    Definitions.NeckName,
                ),
                lambda x: MotionModule(x),
                lambda x: MirrorModule(
                    x, Vector3.Axis.ZPositive, Vector3.Create(0, 0, 180)
                ),
            ],
            max_files=MAX_FILES,
        )

        self.DataSampler = DataSampler(
            self.Dataset,
            framerate=FRAMERATE,
            batch_size=BATCH_SIZE,
            function=self.GetTrainingFeatures,
        )

        self.RootSmoothing = TimeSeries(
            start=-SMOOTHING_WINDOW / 2.0,
            end=SMOOTHING_WINDOW / 2.0,
            samples=RESOLUTION,
        )

        self.ControlSeries = TimeSeries(start=-0.5, end=0.5, samples=RESOLUTION)

        self.Network = Tensor.ToDevice(
            MultiLayerPerceptron.Model(
                input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=2048
            )
        )

        self.Optimizer = Utility.CosineAnnealingOptimizer(
            self.Network.parameters(),
            self.DataSampler.BatchSize,
            self.DataSampler.SampleCount,
        )

        self.LossHistory = Plotting.LossHistory(
            "Loss History", drawInterval=DRAW_INTERVAL, yScale="log"
        )

        self.Paused = False
        self.Trainer = self.Training()

    def Standalone(self):
        self.Editor = AI4Animation.Scene.AddEntity("Trainer").AddComponent(
            MotionEditor,
            self.Dataset,
            os.path.join(ASSETS_PATH, "Model.glb"),
            Definitions.FULL_BODY_NAMES,
        )
        self.Simulated = AI4Animation.Scene.AddEntity("Simulated").AddComponent(
 
