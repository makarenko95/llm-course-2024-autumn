import torch
from unittest import TestCase
from scripts.model import Model
from scripts.trainer import Trainer


class TestTrainer(TestCase):
    def test_trainer(self):
        train_dataset = [torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)]
        eval_dataset = [torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)]
        model = Model(
            vocab_size=6,
            emb_size=8,
            num_layers=1,
            hidden_size=32
        )

        trainer = Trainer(
            model=model,
            n_epochs=128,
            lr=1e-2,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            train_batch_size=1,
            eval_batch_size=1,
            eval_steps=16
        )
        trainer.train()
        self.assertTrue(trainer.evaluate() < 0.01)

        model.eval()
        logits, _ = model(torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long))
        self.assertEqual(
            logits.argmax(-1)[0].cpu().tolist(),
            [1, 2, 3, 4, 5]
        )
