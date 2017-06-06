import train

train.trainer(data='f8k', dim_image=4096, lrate=0.0001, encoder='lstm', max_epochs=1000,
              dim=300, maxlen_w=150, dispFreq=10, saveto='vse/f8k.npz')