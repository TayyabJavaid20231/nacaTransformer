
time = []
with open('/local/disk1/tjavaid/Plots/plots4STD2_128x128HiddenSize150/loss_raw.txt', 'r') as file:
    for line in  file:
        time.append(float(line[:line.find(',')]))
print(time)
print(sum(time))

