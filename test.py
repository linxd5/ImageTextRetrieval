import train

data = 'f30k'
saveto = 'vse/%s' %data
dim_image = 2048 if data == 'arch' or data == 'arch_small' else 4096


train.trainer(data=data, dim_image=dim_image, lrate=0.01, encoder='bow', max_epochs=100000,
              dim=300, maxlen_w=150, dispFreq=10, validFreq=1000, concat=False, saveto=saveto)