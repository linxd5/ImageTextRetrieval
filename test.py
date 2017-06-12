import train

data = 'arch_small'
saveto = 'vse/%s' %data
dim_image = 2048 if data == 'arch' or data == 'arch_small' else 4096


train.trainer(data=data, dim_image=dim_image, lrate=0.01, encoder='bow', max_epochs=100000,
              dim=300, maxlen_w=100, dispFreq=10, validFreq=10000, concat=False, saveto=saveto)