import train

data = 'arch'
saveto = 'vse/%s' %data
dim_image = 2048 if data == 'arch' or data == 'arch_small' else 4096


train.trainer(data=data, dim_image=dim_image, lrate=0.0001, encoder='lstm', max_epochs=100000,
              dim=300, maxlen_w=150, dispFreq=10, validFreq=10000, saveto=saveto)