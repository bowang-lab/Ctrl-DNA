import itertools
import json
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.distributions as td
import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics import Accuracy
from transformers import AutoModelForCausalLM

HYENADNA_MODEL_NAMES = [
    "hyenadna-tiny-16k-seqlen-d128",
    "hyenadna-large-1m-seqlen",
    "hyenadna-medium-160k-seqlen",
    "hyenadna-medium-450k-seqlen",
    "hyenadna-small-32k-seqlen",
    "hyenadna-tiny-1k-seqlen",
    "hyenadna-tiny-1k-seqlen-d256",
]


def load_pretrained_model(
    ckpt_dir="./checkpoints/",
    model="hyenadna-medium-160k-seqlen",
    hyenadna_path="/code/hyena-dna",
):
    """
    Load a pretrained hyenaDNA foundation model.

    Args:
        ckpt_dir (str): Path to directory containing downloaded model checkpoints,
            or in which they should be downloaded
        model (str): Name of model to load
        hyenadna_path (str): Path to cloned hyenaDNA repository

    Returns:
        model (nn.Module): pre-trained HyenaDNA foundation model
    """
    sys.path.append(hyenadna_path)
    from src.models.sequence.long_conv_lm import ConvLMHeadModel

    # Check model name
    assert model in HYENADNA_MODEL_NAMES

    # Make directory if needed
    if not os.path.exists(ckpt_dir):
        print("Making checkpoint directory")
        os.makedirs(ckpt_dir)

    # Download model if not already downloaded
    if not os.path.exists(os.path.join(ckpt_dir, "config.json")):
        print("Downloading model")
        config = f"https://huggingface.co/LongSafari/{model}/resolve/main/config.json"
        ckpt = f"https://huggingface.co/LongSafari/{model}/resolve/main/weights.ckpt"
        os.system(f"wget -P {ckpt_dir} {config}")
        os.system(f"wget -P {ckpt_dir} {ckpt}")

    # Load config
    config = json.load(open(os.path.join(ckpt_dir, "config.json"), "r"))

    # Generate model
    model = ConvLMHeadModel(**config)

    # Load weights
    state_dict = torch.load(
        os.path.join(ckpt_dir, "weights.ckpt"), map_location=torch.device("cpu")
    )
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        state_dict["state_dict"], "model."
    )
    model_state_dict = state_dict["state_dict"]
    for key in list(model_state_dict.keys()):
        if "torchmetrics" in key:
            model_state_dict.pop(key)

    model.load_state_dict(model_state_dict)
    return model


class LightningModel(pl.LightningModule):
    """
    LightningModule class to train and use autoregressive token-conditioned
    regLM language models.

    Args:
        config (dict): Config dictionary containing model parameters
        ckpt_dir (str): Path to directory containing downloaded model checkpoints,
            or in which they should be downloaded
        hyenadna_path (str): Path to cloned hyenaDNA repository
        save_dir (str): Directory to save model checkpoints and logs
        lr (float): Learning rate
        label_len (int): Number of label tokens preceding each DNA sequence
    """

    def __init__(
        self,
        config=None,
        ckpt_dir="./checkpoints/hyenadna-medium-160k-seqlen",
        hyenadna_path="/code/hyena-dna",
        save_dir=".",
        lr=1e-4,
        label_len=None,
    ):
        super().__init__()

        self.save_dir = save_dir
        self.label_len = label_len
        self.save_hyperparameters(ignore=["model"])
        self.lr = lr

        ckpt_dir = 'LongSafari/hyenadna-medium-160k-seqlen-hf'
        hyenadna_path = "/blob/ICLR/hyena/hyena-dna"
        
        # Build model
        '''
        if ckpt_dir is not None:
            self.model = load_pretrained_model(
                ckpt_dir=ckpt_dir, hyenadna_path=hyenadna_path
            )
        elif config is not None:
            sys.path.append(hyenadna_path)
            from src.models.sequence.long_conv_lm import ConvLMHeadModel

            self.model = ConvLMHeadModel(**config)
        else:
            raise ValueError("either config or ckpt_dir must be provided.")
        '''
        self.model = AutoModelForCausalLM.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        # Print number of model parameters
        self.n_params = sum(p.numel() for p in self.model.parameters())
        print("number of parameters: %.2fM" % (self.n_params / 1e6,))

        # Metrics: accuracy
        self.train_acc = Accuracy(task="multiclass", num_classes=16, ignore_index=0)
        self.val_acc = Accuracy(task="multiclass", num_classes=16, ignore_index=0)

        # Encoding
        self.label_stoi = {
            "0": 2,
            "1": 3,
            "2": 4,
            "3": 5,
            "4": 6,
            "5": 7,
            "6": 8,
            "7": 9,
            "8": 10,
            "9": 11,
        }
        self.base_stoi = {
            "A": 7,
            "C": 8,
            "G": 9,
            "T": 10,
            "N": 11,
        }
        self.label_itos = {v: k for k, v in self.label_stoi.items()}
        self.base_itos = {v: k for k, v in self.base_stoi.items()}

        # Loss function
        # Trailing zeros in the label will be ignored in calculating the loss
        self.loss = lambda logits, y: F.cross_entropy(logits, y, ignore_index=0)
        self.label_len=label_len

    def forward(self, x, drop_label=False, return_logits=False):
        """
        Args:
            x (torch.tensor, dtype torch.float32): tensor of shape (N, L)
            drop_label (bool): Whether to drop the predictions for the
                positions corresponding to label tokens
            return_logits (bool): If true, return logits. Otherwise, return
                probabilities

        Returns:
            logits (torch.tensor, dtype torch.float32): tensor of shape
                (N, 16, L - label_len) if drop_label is True,
                or (N, 16, L) if drop_label is False.
                Note that the prediction for the END token (1) as well as the
                hypothetical position after it will be included.
        """
        #print(self.model(x)[0])
        logits = self.model(x).logits.swapaxes(
            1, 2
        )  # N, label + seq + end + trailing zeros

        # Drop the label probabilities
        if drop_label:
            logits = logits[:, :, self.label_len :]  # N, seq + end + trailing


        mask = torch.full_like(logits, fill_value=-1e8)
        mask[:, [7, 8, 9, 10]] = logits[:, [7, 8, 9, 10]]
        
        # Return logits or normalized probabilities
        if return_logits:
            # return logits
            return mask
        else:
            # return logits.softmax(1)
            return mask.softmax(1)

    def get_data(
            self,
            labels,
            max_length,
    ):
        if isinstance(labels, str):
            labels = [labels]
        
        assert len(labels[0]) == self.label_len
        # bases to add
        if max_length is None:
            max_length = self.seq_len
        batch_size = len(labels)
        # Encode labels
        idxs = self.encode_labels(labels, add_start=True).to(
            self.device
        )  # N, label_len+1
        #print(idxs)
        len_bos = 1 + len(labels[0])

        obs = torch.zeros(max_length + len_bos, batch_size, dtype=torch.long).to(self.device)
        obs[:len_bos] = idxs.T

        nonterms = torch.zeros(max_length + len_bos, batch_size, dtype=torch.bool).to(self.device)
        rewards = torch.zeros(max_length + len_bos -1, batch_size, dtype=torch.float32).to(self.device) # only the last 
        end_flags = torch.zeros(1, batch_size, dtype=torch.bool).to(self.device)

        for i in range(len_bos, max_length + len_bos):
            logits = self.forward(obs[:i].T, return_logits=True)[..., -1]
            
            preds_dist = td.Categorical(logits=logits)
            preds = preds_dist.sample()
            obs[i] = preds
            nonterms[i-1] = ~end_flags

            EOS_sampled = (preds == 100) # just a placeholder which cannot be sampled
            rewards[i-1] = EOS_sampled * (~end_flags)

            end_flags = torch.ge(end_flags + EOS_sampled, 1)
            if torch.prod(end_flags) == 1:
                break
        
        # if i == max_length + len_bos - 1:
        #     rewards[-1] = rewards[-1] + (~end_flags) # this is nonsense because the last position is always the end token

        # assert rewards.sum() == batch_size
        assert rewards.sum() == 0
        obs = obs[:i+1]
        # obs = obs[len_bos:]
        nonterms = nonterms[:i+1]
        # nonterms = nonterms[len_bos:]
        # rewards = rewards[:i]
        episode_lens = nonterms.sum(0).cpu()
    
        # assert rewards is all 0
        
        return obs, rewards, nonterms, episode_lens
                

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(
            x, drop_label=True, return_logits=True
        )  # N, seq + end + trailing
        loss = self.loss(logits, y)  # Loss will be calculated over seq + end positions
        self.log(
            "train_loss",
            loss,
            logger=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(
            x, drop_label=True, return_logits=True
        )  # N, seq + end + trailing
        loss = self.loss(logits, y)  # Loss will be calculated over seq + end positions
        self.val_acc.update(logits.argmax(1), y)
        self.log(
            "val_loss",
            loss,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_epoch_end(self, output):
        val_acc = self.val_acc.compute()
        self.log("val_acc", val_acc)
        val_loss = torch.mean(torch.Tensor(output))
        print(f"\nVal loss: {val_loss}, val acc: {val_acc}")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=float(self.lr))
        return optimizer

    def train_on_dataset(
        self,
        train_dataset,
        val_dataset,
        batch_size=128,
        num_workers=8,
        device=0,
        max_epochs=3,
        val_check_interval=5000,
        weights=None,
        save_all=False,
    ):
        """
        Train regLM model.

        Args:
            train_dataset (CharDataset): Training dataset
            val_dataset (CharDataset): Validation dataset
            batch_size (int): Batch size
            num_workers (int): Number of workers for training
            device (int): GPU index
            max_epochs (int): Number of epochs to train
            val_check_interval (int): Number of steps after which to
                check validation loss

        Returns:
            pl.Trainer object
        """
        torch.set_float32_matmul_precision("medium")

        # Save dataset params
        self.seq_len = train_dataset.seq_len
        self.label_len = train_dataset.label_len
        self.unique_labels = train_dataset.unique_labels

        # Logger
        logger = CSVLogger(self.save_dir)

        # Create callback
        callbacks = [
            ModelCheckpoint(monitor="val_acc", mode="max"),
            ModelCheckpoint(monitor="val_loss", mode="min"),
        ]
        if save_all:
            callbacks.append(ModelCheckpoint(every_n_epochs=1, save_top_k=-1))

        # Set up trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu",
            devices=[device],
            logger=logger,
            callbacks=callbacks,
            default_root_dir=self.save_dir,
            val_check_interval=val_check_interval,
        )

        # Make dataloaders
        if weights is None:
            train_data = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
        else:
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            )
            train_data = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                sampler=sampler,
            )

        val_data = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        # First validation pass
        trainer.validate(model=self, dataloaders=val_data)

        # Training
        trainer.fit(model=self, train_dataloaders=train_data, val_dataloaders=val_data)

        return trainer

    def compute_accuracy_on_dataset(
        self,
        dataset,
        batch_size=64,
        num_workers=8,
    ):
        """
        Perform inference on a dataset and return per-example accuracy
        Note: this will include the accuracy of predicting the END token (1)

        Args:
            dataset (CharDataset): Inference dataset
            batch_size (int): Batch size for inference
            num_workers (int): Number of workers for inference

        Returns: List of booleans indicating whether the predicted base at each position
        was equal to the true label or not.
        """
        # Make dataloader
        dl = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Empty lists to hold predictions
        y_hat = []
        y = []

        self.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(iter(dl)):
                x = batch[0].to(self.device)
                probs = self.forward(
                    x, drop_label=True
                ).squeeze()  # N, 16, seq+end+trailing
                y_hat.append(probs.argmax(1).cpu().detach())  # N, seq+end+trailing
                y.append(batch[1].detach().squeeze())  # N, seq+end+trailing zeros

        # Stack batched predictions
        y_hat = torch.vstack(y_hat).numpy()  # N, L
        y = torch.vstack(y).numpy()  # N, L

        # Compare predicted base and true label where the true label is not 0.
        # Hence, this computation will include the END token (1) but not any
        # hypothetical bases / trailing zeros.
        is_non_zero = y != 0
        is_equal = y_hat == y

        # Return booleans
        return [e[n] for e, n in zip(is_equal, is_non_zero)]

    def encode_labels(self, labels, add_start=False):
        """
        Encode labels as a list of indices for inference

        Args:
            labels (list, str): Strings of label tokens
            add_start (bool): Whether to add the start token (0)

        Returns:
            idxs (torch.LongTensor): tensor of shape (N, L)
        """
        if isinstance(labels, str):
            labels = [labels]

        # Convert label to indices
        #print(self.label_stoi)
        idxs = torch.LongTensor(
            [[self.label_stoi[tok] for tok in label] for label in labels]
        )  # N, label_len
        #idxs=idxs.unsqueeze(1)
        # Add start token
        if add_start:
            idxs = torch.cat(
                (torch.LongTensor([[0]] * idxs.shape[0]), idxs), axis=1
            )  # N, 1 + label_len
        return idxs

    def encode_seqs(self, seqs, add_stop=False):
        """
        Encode sequences as lists of indices for inference

        Args:
            seqs (list, str): DNA sequence(s) as string or list of strings
            add_stop (bool): Whether to add an end token (1) after each sequence

        Returns:
            idxs (torch.LongTensor): tensor of shape (N, L) if add_stop is False
            or (N, L+1) if add_stop is True.
        """
        if isinstance(seqs, str):
            seqs = [seqs]

        # Convert sequences to indices
        idxs = torch.LongTensor(
            [[self.base_stoi[tok] for tok in seq] for seq in seqs]
        )  # N, seq_len

        # Add END token
        if add_stop:
            idxs = torch.cat(
                (idxs, torch.LongTensor([[1]] * idxs.shape[0])), axis=1
            )  # N, seq_len+1

        return idxs

    def encode(self, seqs, labels, add_start=False, add_stop=False):
        """
        Encode sequences and labels as indices for model inference.

        Args:
            seqs (list, str): Strings of base tokens
            label (list, str): Strings of label tokens
            add_start (bool): Whether to add the start token (0)
            add_stop (bool): Add an end token (1) after the sequence

        Returns:
            idxs (torch.LongTensor): tensor of shape (N, L)
        """
        
        return torch.cat(
            [
                self.encode_labels(labels, add_start=add_start),
                self.encode_seqs(seqs, add_stop=add_stop),
            ],
            axis=1,
        )

    def decode(self, idxs, ignore_num=0):
        """
        Decodes indices into DNA sequences

        Args:
            idxs (torch.LongTensor): tensor or array of shape (N, L)

        Returns:
            seqs (list): list of strings
        """
        # if torch 
        if isinstance(idxs, torch.Tensor):
            if idxs.dim() == 1:  # L
                idxs = idxs.unsqueeze(0)  # 1, L

            idxs = idxs.cpu().detach().numpy()
        elif isinstance(idxs, np.ndarray):
            if idxs.ndim == 1:  # L
                idxs = idxs[np.newaxis, :]
        idxs = idxs[:, ignore_num:]
        # Create empty list
        seqs = []

        # Fill list
        for x in idxs:
            curr_seq = []

            # Decode
            for pos, ix in enumerate(x):
                # Ignore start token
                if (pos == 0) and (ix == 0):
                    continue
                        
                # Terminate at end token
                if ix == 1:
                    break

                # Replace non-bases with N
                elif (ix < 7) or (ix > 11):
                    ix = 11

                curr_seq.append(self.base_itos[ix])

            seqs.append("".join(curr_seq))
        return seqs

    def probs_to_likelihood(self, probs, idxs):
        """
        Compute the likelihood of each base in a sequence given model predictions
        on the sequence.

        Args:
            probs (torch.FloatTensor): tensor of shape (N, 16, L)
            idxs (torch.LongTensor): tensor of shape (N, L)

        Returns:
            tensor of shape (N, L) containing the probabilities of real bases
        """
        # Check shapes
        assert probs.dim() == 3, probs.shape
        assert probs.shape[1] == 16, probs.shape
        assert idxs.dim() == 2, idxs.shape
        assert idxs.shape[0] == probs.shape[0], (idxs.shape, probs.shape)
        assert idxs.shape[1] == probs.shape[2], (idxs.shape, probs.shape)

        # Construct mask indicating positions of actual bases
        mask = F.one_hot(idxs, num_classes=16).type(torch.bool)

        # Return probabilities at real bases
        return torch.masked_select(probs.swapaxes(1, 2).cpu().detach(), mask).reshape(
            idxs.shape
        )

    def P_seqs(self, seqs, labels, per_pos=False, log=True):
        """
        Args:
            seqs (list, str): Sequences as strings
            labels(list, str): Labels as strings
            log (bool): Return log likelihood
            include_end (bool): Include the end token

        Returns:
            np.array of shape (N)
        """
        idxs = self.encode(
            seqs, labels, add_start=True, add_stop=True
        )  # N, 0+label+seq+1

        # Compute probabilities
        self.eval()
        probs = self.forward(idxs.to(self.device), drop_label=False)[
            :, :, :-1
        ]  # N, 16, label+seq+1

        # Compute likelihood
        idxs = idxs[:, 1:]  # N, label+seq+1
        L = self.probs_to_likelihood(probs, idxs).numpy()  # N, label+seq+1

        # Log and sum
        if log:
            L = np.log(L)
        if per_pos:
            return L
        else:
            if log:
                return np.sum(L, 1)  # N
            else:
                return np.product(L, 1)  # N

    def P_seqs_given_labels(self, seqs, labels, per_pos=False, log=True, add_stop=True):
        """
        Args:
            seqs (list, str): Sequences as strings
            labels(list, str): Labels as strings
            log (bool): Return log likelihood
            include_end (bool): Include the end token

        Returns:
            np.array of shape (N)
        """
        # Encode sequence with labels
        idxs = self.encode(
            seqs, labels, add_start=True, add_stop=add_stop
        )  # N, 0+label+seq OR N, 0+label+seq+1

        # Compute probabilities
        self.eval()
        probs = self.forward(idxs.to(self.device), drop_label=True)[
            :, :, :-1
        ]  # N, 16, seq OR N, 16, seq+1
        idxs = idxs[:, 1 + self.label_len :]  # N, seq

        # Compute likelihoods
        L = self.probs_to_likelihood(probs=probs, idxs=idxs).numpy()  # N, seq

        # Log and sum
        if log:
            L = np.log(L)
        if per_pos:
            return L
        else:
            if log:
                return np.sum(L, 1)  # N
            else:
                return np.product(L, 1)  # N

    def P_labels_given_seqs(self, seqs, labels, per_pos=True, log=True):
        # List all possible labels
        possible_labels = [
            "".join(x)
            for x in itertools.product(self.unique_labels, repeat=self.label_len)
        ]
        # Compute likelihood for each possible label
        likelihoods = np.stack(
            [
                self.P_seqs(
                    seqs,
                    [label] * len(seqs),
                    per_pos=True,
                    log=False,
                )
                for label in possible_labels
            ],
            axis=-1,
        )  # N, L, possible_labels
        # Calculate numerator and denominator for posterior
        denominators = np.sum(likelihoods, -1)  # N, L
        numerators = np.vstack(
            [
                likelihoods[i, :, np.array(possible_labels) == labels[i]]
                for i in range(len(seqs))
            ]
        )

        # Compute posterior
        if log:
            P = np.log(numerators) - np.log(denominators)  # N, L
        else:
            P = np.divide(numerators, denominators)  # N, L
        if per_pos:
            return P
        else:
            if log:
                return np.sum(P, 1)  # N
            else:
                return np.product(P, 1)  # N

    def normalize_filtered_probs(self, filtered_probs):
        """
        Normalize probabilities at each position to sum to 1.

        Args:
            filtered_probs (torch.floatTensor): Tensor of shape (N, 16, L) or (N, 16)

        Returns:
            Normalized tensor of the same shape
        """
        return filtered_probs / filtered_probs.sum(dim=1, keepdim=True)

    def filter_base_probs(self, probs, normalize=True):
        """
        Return probabilities for valid bases only

        Args:
            probs (torch.tensor, dtype torch.float32): tensor of shape (N, 16)
            normalize (bool): Whether to re-normalize the probabilities at each
            position to sum to 1.

        Returns:
            filtered_probs (torch.FloatTensor): tensor of shape (N, 4)
        """
        # Check shape
        assert probs.dim() == 2
        assert probs.shape[1] == 16

        # Filter probabilities
        filtered_probs = probs[:, [7, 8, 9, 10]]

        # Normalize
        if normalize:
            filtered_probs = self.normalize_filtered_probs(filtered_probs)
        return filtered_probs

    def threshold_probs(self, filtered_probs, top_k=None, top_p=None):
        """
        Threshold the filtered probabilities for valid bases

        Args:
            filtered_probs (torch.tensor, dtype torch.float32): tensor of shape (N, 4)
            top_k (int): Select the top k bases at each position. Set probabilites
                of other bases to 0.
            top_p (float): Select the top bases at each position until their cumulative
                probability reaches this value. Set probabilites of other bases to 0.

        Returns:
            tensor of shape (N, 4)
        """
        # Check shape
        assert filtered_probs.dim() == 2
        assert filtered_probs.shape[1] == 4

        # Top K sampling
        if top_k is not None:
            p_idxs = filtered_probs.argsort(1, descending=True)
            for seq_idx in range(filtered_probs.shape[0]):
                filtered_probs[seq_idx, p_idxs[seq_idx, top_k:]] = 0

        # Top P (nucleus) sampling
        if top_p is not None:
            p_sorted, p_idxs = filtered_probs.sort(1, descending=True)
            cut = (p_sorted.cumsum(1) > top_p).cpu().detach().numpy().argmax(1).tolist()
            for seq_idx, cut_idx in enumerate(cut):
                if cut_idx < 3:
                    filtered_probs[seq_idx, p_idxs[seq_idx, cut_idx + 1 :]] = 0

        return filtered_probs

    def sample_idxs(
        self, probs, random_state=None, top_k=None, top_p=None, normalize_filtered=True
    ):
        """
        Sample from model predictions at a single position to return a single
        base per example

        Args:
            probs (torch.tensor, dtype torch.float32): tensor of shape (N, 16)
            random_state (torch.Generator): torch.Generator object
            top_k (int): Select the top k bases at each position. Set probabilites
                of other bases to 0.
            top_p (float): Select the top bases at each position until their cumulative
                probability reaches this value. Set probabilites of other bases to 0.
            normalize_filtered (bool): Normalize probabilities to sum to 1
                after filtering

        Returns:
            idxs (torch.LongTensor): tensor of shape (N)
        """
        # Check
        assert probs.dim() == 2, probs.shape
        assert probs.shape[1] == 16, probs.shape

        # Subset to valid bases
        probs = self.filter_base_probs(probs, normalize=normalize_filtered)  # N, 4

        # Threshold probabilities for sampling
        probs = self.threshold_probs(probs, top_k=top_k, top_p=top_p)  # N, 4

        # Re-normalize
        probs = self.normalize_filtered_probs(probs)

        # Sample
        if random_state is None:
            idxs = probs.multinomial(1).squeeze() + 7
        else:
            idxs = probs.multinomial(1, generator=random_state).squeeze() + 7

        # Send to device
        return idxs.to(probs.device)

    @torch.no_grad()
    def generate(
        self,
        labels,
        max_new_tokens=None,
        temperature=1.0,
        top_k=None,
        top_p=None,
        normalize_filtered=True,
        seed=None,
    ):
        """
        Args:
            labels (str, list): Strings of label tokens
            max_new_tokens (int): Maximum number of tokens to add
            temperature (float): Temperature
            top_k (int): Select the top k bases at each position. Set probabilites
                of other bases to 0.
            top_p (float): Select the top bases at each position until their cumulative
                probability reaches this value. Set probabilites of other bases to 0.
            normalize_filtered (bool): Normalize probabilities to sum to 1
                after filtering
            seed (int): Random seed for sampling

        Returns:
            seqs (list): List of strings
        """
        # Check labels
        if isinstance(labels, str):
            labels = [labels]
        assert len(labels[0]) == self.label_len

        # bases to add
        if max_new_tokens is None:
            max_new_tokens = self.seq_len

        # Encode labels
        idxs = self.encode_labels(labels, add_start=True).to(
            self.device
        )  # N, label_len+1

        # Get random state
        rng = torch.Generator(device=self.device)
        if seed is not None:
            rng.manual_seed(seed)

        # Add bases
        for _ in range(max_new_tokens):
            self.eval()

            # Predict next base probabilities
            probs_next = self.forward(idxs, return_logits=False)[:, :, -1]  # N, 16

            # Get next indices
            idxs_next = self.sample_idxs(
                probs_next,
                random_state=rng,
                top_k=top_k,
                top_p=top_p,
                normalize_filtered=normalize_filtered,
            )  # N

            # Add new indices
            idxs = torch.cat((idxs, idxs_next.unsqueeze(1)), dim=1)

        return self.decode(idxs[:, self.label_len + 1 :])

    def beam_search(
        self, beam_width, batch_size, label, random_state=None, sample=False
    ):
        bases = self.encode_seqs(["ACGT"]).T
        idxs = self.encode_labels([label], add_start=True)
        self.eval()

        for i in tqdm.tqdm(range(self.seq_len)):
            # Construct all possible sequences
            idxs = idxs.repeat_interleave(4, 0)
            bases_ = bases.tile(idxs.shape[0] // 4, 1)
            possibilities = torch.hstack((idxs, bases_))
            probs = []

            # Compute likelihood of each sequence
            with torch.no_grad():
                for st in range(0, possibilities.shape[0], batch_size):
                    en = min(st + batch_size, possibilities.shape[0])
                    batch = possibilities[st:en].to(self.device)
                    batch_probs = self.forward(batch, drop_label=True)[:, :, :-1].cpu()
                    probs.append(batch_probs)

            probs = torch.cat(probs)
            L = self.probs_to_likelihood(
                probs=probs, idxs=possibilities[:, self.label_len + 1 :]
            )
            L = torch.sum(torch.log(L), 1)

            # Sample sequences for next iteration
            curr_beam_width = min(beam_width, len(L))
            if sample:
                L = torch.exp(L)
                if random_state is None:
                    choice = L.multinomial(curr_beam_width).squeeze()
                else:
                    choice = L.multinomial(
                        curr_beam_width, generator=random_state
                    ).squeeze()
            else:
                choice = L.numpy().argsort()[-curr_beam_width:]
            idxs = possibilities[choice, :]

        # Decode
        seqs = self.decode(idxs[:, self.label_len + 1 :])
        return seqs

    def on_save_checkpoint(self, checkpoint):
        """
        Save data relevant parameters to the model checkpoint on training.
        """
        checkpoint["hyper_parameters"]["seq_len"] = self.seq_len
        checkpoint["hyper_parameters"]["label_len"] = self.label_len
        checkpoint["hyper_parameters"]["unique_labels"] = self.unique_labels

    def get_likelihood(self, obs, nonterms):
        logits = self.forward(obs[:-1].T, drop_label=False, return_logits=True)
        # (bs, vocab, len) -> (len, bs, vocab)
        bs, vocab_size, seq_len = logits.size()
        logits = logits.permute(2, 0, 1)
        dist = td.Categorical(logits=logits)
        logprobs = dist.log_prob(obs[1:]) * nonterms[:-1]

        log_of_probs = F.log_softmax(dist.logits, dim=-1) * nonterms[:-1].unsqueeze(-1)
        action_probs = dist.probs * nonterms[:-1].unsqueeze(-1)
        
        return logprobs, log_of_probs, action_probs
    def sequence_log_probs_from_logits(self,logits: torch.tensor, output_ids: torch.tensor):
        log_prob = F.log_softmax(logits, dim=-1)
        return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


    def sequences_log_probs(
        self,
        sequence_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        #print(sequence_ids.shape,attention_mask.shape)
        #print(sequence_ids)
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
        logits = self.forward(
            sequence_ids[:-1].T,drop_label=False, return_logits=True
            
        )
        #print(logits.shape,attention_mask.shape)
        #logits = logits*(attention_mask[:-1].transpose(1,0).unsqueeze(1))
        #print(logits.shape)
        
        log_probs = self.sequence_log_probs_from_logits(
            logits=logits.transpose(-1,-2).to(torch.float32),
            output_ids=sequence_ids[1:].T,
        )*(attention_mask[:-1].transpose(1,0))
        #print(log_probs.shape)
        return log_probs.transpose(-1,-2)
    def get_per_token_logps(self,logits, input_ids):
        
        per_token_logps = [] # Use a loop to reduce memory peak.
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    def get_ref_per_token_logps(self,input_ids):
        input_ids = input_ids.T
        logits = self.forward(input_ids).transpose(-1,-2)  # (B, L, V)
        #print(logits.shape,input_ids.shape)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        #print(logits.shape,input_ids.shape)
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)\
        
    