from network.models import AlexNet, PamapModel, lenet5v, TimeSeriesModel
import copy
from util.config import normalize_dataset_name


def modelsel(args, device):
    dataset = normalize_dataset_name(args.dataset)
    if dataset in ['vlcs', 'pacs', 'officehome', 'office-caltech', 'covid']:
        server_model = AlexNet(num_classes=args.num_classes).to(device)
    elif 'medmnist' in dataset:
        server_model = lenet5v().to(device)
    elif 'pamap' in dataset:
        server_model = PamapModel().to(device)
    elif 'cwru' in dataset or 'seu' in dataset:
        server_model = TimeSeriesModel(out_dim=args.num_classes).to(device)

    client_weights = [1/args.n_clients for _ in range(args.n_clients)]
    models = [copy.deepcopy(server_model).to(device)
              for _ in range(args.n_clients)]
    return server_model, models, client_weights
