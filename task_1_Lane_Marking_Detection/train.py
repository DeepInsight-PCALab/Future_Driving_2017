#!/usr/bin/env python
# coding=utf-8
import time
from options.train_options import TrainOptions                          # finish
from data.custom_dataset_data_loader import CustomDatasetDataLoader     # finish
from models.models         import create_model                          # finish
from util.visualizer       import Visualizer                            # finish
from util.recorder         import Recorder                              # finish

opt           = TrainOptions().parse()
train_loader  = CustomDatasetDataLoader(opt, 'train')
val_loader    = CustomDatasetDataLoader(opt, 'val')
train_dataset = train_loader.load_data()
val_dataset   = val_loader.load_data()

dataset_size  = len(train_loader)
print('#training images = %d' % dataset_size)

model        = create_model(opt)
visualizer   = Visualizer(opt)
recorder     = Recorder()
total_steps  = 0

for epoch in range(opt.epoch_count, opt.schedule_max + 1):
    epoch_start_time = time.time()
    epoch_iter       = 0

    for i, data in enumerate(train_dataset):
        iter_start_time = time.time()
        total_steps    += opt.batchSize
        epoch_iter     += opt.batchSize
        model.set_input_and_label(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(0.5), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t      = (time.time() - iter_start_time) #/ opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            # ignore display_id

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        '''
        recorder.clear()
        for i, data in enumerate(val_dataset):
            model.set_input_and_label(data)
            model.forward()
            recorder.add(model.get_current_errors())
        str_val_error = recorder.summary()
        print('evaluating the model %s at the end of epoch %d, iters %d' % (str_val_error, epoch, total_steps))

        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.save('latest|' + str_val_error)
        model.save('%03d' % epoch + '|' + str_val_error)
        '''
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.save('latest')
        model.save('%03d' % epoch)


    print('End of epoch %d / %d \t Time Taken: %.2f min' % (epoch, opt.schedule_max, (time.time() - epoch_start_time) / 60.0))
    model.update_learning_rate(epoch)
