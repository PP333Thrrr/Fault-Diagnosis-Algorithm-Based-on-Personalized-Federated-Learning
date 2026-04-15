# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import copy

def communication(args, server_model, models, client_weights):
    client_num = len(models)
    server_state = server_model.state_dict()
    model_states = [model.state_dict() for model in models]
    with torch.no_grad():
        if args.alg.lower() == 'fedbn':
            for key in server_state.keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_state[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * model_states[client_idx][key]
                    server_state[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        model_states[client_idx][key].data.copy_(server_state[key])
        elif args.alg.lower()=='fedap':
            tmpmodels = []
            for i in range(client_num):
                tmpmodels.append(copy.deepcopy(models[i]).to(args.device))
            tmpmodel_states = [model.state_dict() for model in tmpmodels]
            with torch.no_grad():
                for cl in range(client_num):
                    for key in server_state.keys():
                        temp = torch.zeros_like(server_state[key], dtype=torch.float32)
                        for client_idx in range(client_num):
                            temp += client_weights[cl,client_idx] * tmpmodel_states[client_idx][key]
                        server_state[key].data.copy_(temp)
                        if 'bn' not in key:
                            model_states[cl][key].data.copy_(server_state[key])
        else:
            for key in server_state.keys():
                if 'num_batches_tracked' in key:
                    server_state[key].data.copy_(model_states[0][key])
                else:
                    temp = torch.zeros_like(server_state[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * model_states[client_idx][key]
                    server_state[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        model_states[client_idx][key].data.copy_(server_state[key])
    return server_model, models
    
