from cellpose import io, models, train
io.logger_setup()

# Set paths for training and test data
train_dir = "data/BF+IF Experiments_2D_train_test_dataset/train"
test_dir = "data/BF+IF Experiments_2D_train_test_dataset/test"

# Load training and test data
output = io.load_train_test_data(train_dir, test_dir, 
                                image_filter="_BF", # Filter for brightfield images
                                mask_filter="_Cells", # Filter for mask images
                                look_one_level_down=True)

images, labels, image_names, test_images, test_labels, image_names_test = output

# Initialize model for brightfield images
model = models.CellposeModel(gpu=False, # Don't use GPU
                            model_type='cyto2') # Base model for brightfield

# Train the model
model_path, train_losses, test_losses = train.train_seg(model.net,
                            train_data=images, train_labels=labels,
                            channels=[0,0], normalize=True,
                            test_data=test_images, test_labels=test_labels,
                            weight_decay=1e-4, SGD=True, learning_rate=0.1,
                            n_epochs=100, model_name="cellpose_brightfield_model")

# model.train(images, labels,
#            test_images, test_labels,
#            channels=[0,0], # Single grayscale channel for brightfield
#            train_files=image_names,
#            test_files=image_names_test,
#            learning_rate=0.2,
#            weight_decay=0.00001, 
#            n_epochs=500,
#            batch_size=8,
#            min_train_masks=5,
#            model_name='cellpose_brightfield_model')

