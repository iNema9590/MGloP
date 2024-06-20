from image_explainer import *
import matplotlib.pyplot as plt
from modules import *
import glob
import os
import torch.utils.data as data
from captum.influence import TracInCP, TracInCPFast, TracInCPFastRandProj

import networkx as nx
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

embeds = load_inception_embeds()


class GenericDataAttribution:
    def __init__(self, model, trainset, testset):
        self.model = model
        self.trainset = trainset
        self.testset = testset

    def print_explanation(self, N, test_index):

        new_line = '\n'
        fig, axs = plt.subplots(3, 5, figsize=(18, 8))
        if test_index > 600:
            axs[0, 0].imshow(captioned_image(self.model, embeds, 'train', test_index - 600))
        else:
            axs[0, 0].imshow(captioned_image(self.model, embeds, 'test', test_index))
        axs[0, 0].axis('off')

        exlist = self.explain(N, test_index)
        for i, j in enumerate(exlist):
            if i < 5:
                axs[1, i].imshow(captioned_image(self.model, embeds, 'train', j))
                axs[1, i].axis('off')
                axs[0, i].axis('off')
            else:
                axs[2, i - 5].imshow(captioned_image(self.model, embeds, 'train', j))
                axs[2, i - 5].axis('off')
                axs[0, i - 5].axis('off')

        plt.show()

    def explain(self, N, test_index):
        pass

    def Xgraph(self, N):
        class_name = type(self).__name__

        influential_samples = {t: self.explain(N, t) for t in tqdm(range(len(self.testset)))}
        G = nx.Graph()
        for i in range(len(self.testset)):
            G.add_node(i, bipartite=0, embedding=self.testset[:][0][i].numpy())
        for point, samples in influential_samples.items():
            for sample in samples:
                G.add_node(f'ex-{sample}', bipartite=1, embedding=self.trainset[:][0][sample].numpy())
                G.add_edge(point, f'ex-{sample}')
        nx.write_gpickle(G, f'/data/ikhtiyor/bigraph_both_{2 * N}_{class_name}.pkl')


class InfluenceFuntion(GenericDataAttribution):
    def explain(self, N, test_index):
        module = LiSSAInfluenceModule(
            model=self.model,
            objective=BinClassObjective(),
            train_loader=data.DataLoader(self.trainset, batch_size=32),
            test_loader=data.DataLoader(self.testset, batch_size=32),
            device=DEVICE,
            damp=0.001,
            repeat=1,
            depth=1800,
            scale=10, )
        influence = module.influences(train_idxs=list(range(len(self.trainset))), test_idxs=[test_index]).argsort(
            descending=True)

        #         if self.model(self.testset[:][0][test_index].unsqueeze(0)).floor()==self.testset[:][1][test_index]:
        #             return module.influences(train_idxs=list(range(len(self.trainset))), test_idxs=[test_index]).argsort(descending=True)[:N].tolist()
        #         else:
        return influence[:N].tolist() + influence[-N:].tolist()


class RelatIF(GenericDataAttribution):
    def explain(self, N, test_index):
        module = LiSSAInfluenceModule(
            model=self.model,
            objective=BinClassObjective(),
            train_loader=data.DataLoader(self.trainset, batch_size=32),
            test_loader=data.DataLoader(self.testset, batch_size=32),
            device=DEVICE,
            damp=0.001,
            repeat=1,
            depth=1800,
            scale=10, )
        influences = module.influences(train_idxs=list(range(len(self.trainset))), test_idxs=[test_index])

        log_loss = np.array([F.binary_cross_entropy(self.model(self.trainset[:][0][i].unsqueeze(0)).squeeze(),
                                                    self.trainset[:][1][i]).item() for i in range(len(self.trainset))])
        log_loss = np.clip(log_loss, 1e-7, 1 - 1e-7)
        relatif = influences / log_loss
        #         if self.model(self.trainset[:][0][test_index].unsqueeze(0)).floor()==self.testset[:][1][test_index]:
        #             return relatif.argsort(descending=True)[:N].tolist()
        #         else:
        return (-relatif).argsort()[:N].tolist() + relatif.argsort()[:N].tolist()


class AIDE(GenericDataAttribution):
    def explain(self, N, test_index):
        pass


class Tracin(GenericDataAttribution):
    def explain(self, N, test_index):
        num_epochs = 26
        test_examples_indices = [test_index]
        test_examples_features = torch.stack([self.testset[i][0] for i in test_examples_indices])
        test_examples_true_labels = torch.Tensor([self.testset[i][1] for i in test_examples_indices]).unsqueeze(1)
        correct_dataset_checkpoints_dir = os.path.join("checkpoints", "dogfish")
        correct_dataset_checkpoint_paths = glob.glob(os.path.join(correct_dataset_checkpoints_dir, "*.pt"))

        def checkpoints_load_func(net, path):
            weights = torch.load(path)
            net.load_state_dict(weights["model_state_dict"])
            return 1.

        correct_dataset_final_checkpoint = os.path.join(correct_dataset_checkpoints_dir,
                                                        "-".join(['checkpoint', str(num_epochs - 1) + '.pt']))
        checkpoints_load_func(self.model, correct_dataset_final_checkpoint)
        tracin_cp_fast = TracInCPFast(
            model=self.model,
            final_fc_layer=list(self.model.children())[-1],
            train_dataset=self.trainset,
            checkpoints=correct_dataset_checkpoint_paths,
            checkpoints_load_func=checkpoints_load_func,
            loss_fn=nn.BCELoss(),
            batch_size=1,
            vectorize=False,
        )
        proponents_indices, proponents_influence_scores = tracin_cp_fast.influence(
            (test_examples_features, test_examples_true_labels), k=N, proponents=True
        )
        opponents_indices, opponents_influence_scores = tracin_cp_fast.influence(
            (test_examples_features, test_examples_true_labels), k=N, proponents=False
        )

        #         if self.model(self.trainset[:][0][test_index].unsqueeze(0)).floor()==self.testset[:][1][test_index]:
        #             return proponents_indices.flatten().tolist()
        #         else:

        return proponents_indices.flatten().tolist() + opponents_indices.flatten().tolist()


class Datamodels(GenericDataAttribution):

    def generate_sample_sets(self, n_sets=1000, fraction=0.3):

        n_samples = int(self.trainset[:][0].shape[0] * fraction)
        sample_sets = [torch.tensor(np.random.choice(self.trainset[:][0].shape[0], size=n_samples, replace=False),
                                    dtype=torch.long) for _ in range(n_sets)]
        return sample_sets

    def train_models(self, sample_sets):
        """
        Train n models on the subsampled sets.
        """
        models = [fit_model(self.trainset[:][0][sample], self.trainset[:][1][sample]) for sample in sample_sets]
        return models

    def explain(self, N, test_index, sample_sets, models):

        dataset = build_dataset(test_index, self.trainset[:][0], self.testset[:][0], models, sample_sets, self.model)
        print(len(dataset))
        lasso_params = train_lasso_model(dataset)
        return (-lasso_params).argsort()[:N].tolist() + lasso_params.argsort()[:N].tolist()

    def Xgraph(self, N):
        class_name = type(self).__name__
        sample_sets = self.generate_sample_sets()
        print("samples are selected")
        models = self.train_models(sample_sets)
        print("data models are trained")
        influential_samples = {t: self.explain(N, t, sample_sets, models) for t in tqdm(range(len(self.testset)))}
        G = nx.Graph()
        for i in range(len(self.testset)):
            G.add_node(i, bipartite=0, embedding=self.testset[:][0][i].numpy())
        for point, samples in influential_samples.items():
            for sample in samples:
                G.add_node(f'ex-{sample}', bipartite=1, embedding=self.trainset[:][0][sample].numpy())
                G.add_edge(point, f'ex-{sample}')
        nx.write_gpickle(G, f'/data/ikhtiyor/bigraph_{N}_{class_name}.pkl')


def generate_sample_sets(X_train, n_sets=1000, fraction=0.3):
    """
    Generate n_sets subsampled sets as proper subsets of X_train.
    """
    n_samples = int(X_train.shape[0] * fraction)
    sample_sets = [torch.tensor(np.random.choice(X_train.shape[0], size=n_samples, replace=False), dtype=torch.long) for
                   _ in range(n_sets)]
    return sample_sets


def train_models(X_train, Y_train, sample_sets):
    """
    Train n models on the subsampled sets.
    """
    models = [fit_model(X_train[sample], Y_train[sample]) for sample in sample_sets]
    return models


def build_dataset(idx, X_train, X_test, models, sample_sets, clf):
    """
    Build the dataset with pairs (Ei, F(X, Si)).
    """
    dataset = []
    #     X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    for i, sample in enumerate(sample_sets):
        Ei = torch.tensor([1 if idx in sample else 0 for idx in range(X_train.shape[0])], dtype=torch.float32)
        p = models[i](X_test[idx].unsqueeze(0)).item()
        q = clf(X_test[idx].unsqueeze(0)).item()
        if p == 1 or q == 1:

            Fi = torch.tensor(abs(p - q), dtype=torch.float32)
            dataset.append((Ei, Fi))
        else:
            Fi = torch.tensor((np.log(q / (1 - q)) - np.log(p / (1 - p))), dtype=torch.float32)
            dataset.append((Ei, Fi))

    return dataset


def train_lasso_model(dataset):
    """
    Train a Lasso model on the generated dataset and return the parameters.
    """
    X = np.array([pair[0].tolist() for pair in dataset])
    y = np.array([pair[1].item() for pair in dataset])

    scaler = StandardScaler()
    #     X_scaled = scaler.fit_transform(X.reshape(-1,1))
    lasso_model = Lasso(alpha=0.005)  # You can adjust the alpha parameter
    print(f"hsape of X: {X.shape}, shape of y: {y.shape}")

    lasso_model.fit(X, y)
    return lasso_model.coef_