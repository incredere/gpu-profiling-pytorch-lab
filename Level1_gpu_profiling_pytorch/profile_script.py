import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

BATCH_SIZE = 256
EPOCHS = 1
LOG_DIR = "./log"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

transform = transforms.Compose([transforms.ToTensor()])

dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

schedule_config = schedule(wait=1, warmup=1, active=3)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule_config,
    on_trace_ready=tensorboard_trace_handler(LOG_DIR),
    record_shapes=True,
    profile_memory=True
) as prof:

    for step, (x, y) in enumerate(loader):

        with record_function("data_transfer"):
            x = x.to(device)
            y = y.to(device)

        with record_function("forward"):
            pred = model(x)
            loss = loss_fn(pred, y)

        with record_function("backward"):
            loss.backward()

        with record_function("optimizer"):
            optimizer.step()
            optimizer.zero_grad()

        prof.step()

        if step > 20:
            break

print("Profiling completed")
