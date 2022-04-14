library(keras)
library(reticulate)
library(tidyverse)
library(tensorflow)
original_dataset_dir <- "archive" # we will only use the labelled data
base_dir <- "." # to store a sebset of data that we are going to use
train_dir <- file.path(base_dir, "train")
dir.create(train_dir)
validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)
test_dir <- file.path(base_dir, "test")
dir.create(test_dir)

train_buoy_dir <- file.path(train_dir, "buoy")
dir.create(train_buoy_dir)
train_cruise_dir <- file.path(train_dir, "cruise ship")
dir.create(train_cruise_dir)
train_ferry_dir <- file.path(train_dir, "ferry boat")
dir.create(train_ferry_dir)
train_freight_dir <- file.path(train_dir, "freight boat")
dir.create(train_freight_dir)
train_gondola_dir <- file.path(train_dir, "gondola")
dir.create(train_gondola_dir)
train_inflatable_dir <- file.path(train_dir, "inflatable boat")
dir.create(train_inflatable_dir)
train_kayak_dir <- file.path(train_dir, "kayak")
dir.create(train_kayak_dir)
train_paper_dir <- file.path(train_dir, "paper boat")
dir.create(train_paper_dir)
train_sailboat_dir <- file.path(train_dir, "sailboat")
dir.create(train_sailboat_dir)

validation_buoy_dir <- file.path(validation_dir, "buoy")
dir.create(validation_buoy_dir)
validation_cruise_dir <- file.path(validation_dir, "cruise ship")
dir.create(validation_cruise_dir)
validation_ferry_dir <- file.path(validation_dir, "ferry boat")
dir.create(validation_ferry_dir)
validation_freight_dir <- file.path(validation_dir, "freight boat")
dir.create(validation_freight_dir)
validation_gondola_dir <- file.path(validation_dir, "gondola")
dir.create(validation_gondola_dir)
validation_inflatable_dir <- file.path(validation_dir, "inflatable boat")
dir.create(validation_inflatable_dir)
validation_kayak_dir <- file.path(validation_dir, "kayak")
dir.create(validation_kayak_dir)
validation_paper_dir <- file.path(validation_dir, "paper boat")
dir.create(validation_paper_dir)
validation_sailboat_dir <- file.path(validation_dir, "sailboat")
dir.create(validation_sailboat_dir)




test_buoy_dir <- file.path(test_dir, "buoy")
dir.create(test_buoy_dir)
test_cruise_dir <- file.path(test_dir, "cruise ship")
dir.create(test_cruise_dir)
test_ferry_dir <- file.path(test_dir, "ferry boat")
dir.create(test_ferry_dir)
test_freight_dir <- file.path(test_dir, "freight boat")
dir.create(test_freight_dir)
test_gondola_dir <- file.path(test_dir, "gondola")
dir.create(test_gondola_dir)
test_inflatable_dir <- file.path(test_dir, "inflatable boat")
dir.create(test_inflatable_dir)
test_kayak_dir <- file.path(test_dir, "kayak")
dir.create(test_kayak_dir)
test_paper_dir <- file.path(test_dir, "paper boat")
dir.create(test_paper_dir)
test_sailboat_dir <- file.path(test_dir, "sailboat")
dir.create(test_sailboat_dir)


folder_list <- list.files("archive/")
folder_path <- paste0("archive/", folder_list, "/")
# Get file name
file_name <- map(folder_path, function(x) paste0(x, list.files(x)))


length(file_name[[2]])
file_train_buoy <- file_name[[2]][1:34] %>% unlist()
file_train_buoy
file.copy(file_train_buoy, file.path(train_buoy_dir))
file_validation_buoy <- file_name[[2]][35:55] %>% unlist()
file_validation_buoy
file.copy(file_validation_buoy, file.path(validation_buoy_dir))
file_test_buoy <- file_name[[2]][56:68] %>% unlist()
file_test_buoy
file.copy(file_test_buoy, file.path(test_buoy_dir))



length(file_name[[3]])
file_train_cruise <- file_name[[3]][1:119] %>% unlist()
file_train_cruise
file.copy(file_train_cruise, file.path(train_cruise_dir))
file_validation_cruise <- file_name[[3]][120:198] %>% unlist()
file_validation_cruise
file.copy(file_validation_cruise, file.path(validation_cruise_dir))
file_test_cruise <- file_name[[3]][199:239] %>% unlist()
file_test_cruise
file.copy(file_test_cruise, file.path(test_cruise_dir))

length(file_name[[4]])
file_train_ferry <- file_name[[4]][1:40] %>% unlist()
file_train_ferry
file.copy(file_train_ferry, file.path(train_ferry_dir))
file_validation_ferry <- file_name[[4]][41:68] %>% unlist()
file_validation_ferry
file.copy(file_validation_ferry, file.path(validation_ferry_dir))
file_test_ferry <- file_name[[4]][69:81] %>% unlist()
file_test_ferry
file.copy(file_test_ferry, file.path(test_ferry_dir))

length(file_name[[5]])
file_train_freight <- file_name[[5]][1:13] %>% unlist()
file_train_freight
file.copy(file_train_freight, file.path(train_freight_dir))
file_validation_freight <- file_name[[5]][14:20] %>% unlist()
file_validation_freight
file.copy(file_validation_freight, file.path(validation_freight_dir))
file_test_freight <- file_name[[5]][21:29] %>% unlist()
file_test_freight
file.copy(file_test_freight, file.path(test_freight_dir))



length(file_name[[6]])
file_train_gondola <- file_name[[6]][1:121] %>% unlist()
file_train_gondola
file.copy(file_train_gondola, file.path(train_gondola_dir))
file_validation_gondola <- file_name[[6]][122:195] %>% unlist()
file_validation_gondola
file.copy(file_validation_gondola, file.path(validation_gondola_dir))
file_test_gondola <- file_name[[6]][196:242] %>% unlist()
file_test_gondola
file.copy(file_test_gondola, file.path(test_gondola_dir))


length(file_name[[7]])
file_train_inflatable <- file_name[[7]][1:10] %>% unlist()
file_train_inflatable
file.copy(file_train_inflatable, file.path(train_inflatable_dir))
file_validation_inflatable <- file_name[[7]][11:15] %>% unlist()
file_validation_inflatable
file.copy(file_validation_inflatable, file.path(validation_inflatable_dir))
file_test_inflatable <- file_name[[7]][16:21] %>% unlist()
file_test_inflatable
file.copy(file_test_inflatable, file.path(test_inflatable_dir))


length(file_name[[8]])
file_train_kayak <- file_name[[8]][1:127] %>% unlist()
file_train_kayak
file.copy(file_train_kayak, file.path(train_kayak_dir))
file_validation_kayak <- file_name[[8]][128:205] %>% unlist()
file_validation_kayak
file.copy(file_validation_kayak, file.path(validation_kayak_dir))
file_test_kayak <- file_name[[8]][205:254] %>% unlist()
file_test_kayak
file.copy(file_test_kayak, file.path(test_kayak_dir))


length(file_name[[9]])
file_train_paper <- file_name[[9]][1:20] %>% unlist()
file_train_paper
file.copy(file_train_paper, file.path(train_paper_dir))
file_validation_paper <- file_name[[9]][21:30] %>% unlist()
file_validation_paper
file.copy(file_validation_paper, file.path(validation_paper_dir))
file_test_paper <- file_name[[9]][31:40] %>% unlist()
file_test_paper
file.copy(file_test_paper, file.path(test_paper_dir))


length(file_name[[10]])
file_train_saiboat <- file_name[[10]][1:244] %>% unlist()
file_train_saiboat
file.copy(file_train_saiboat, file.path(train_sailboat_dir))
file_validation_saiboat <- file_name[[10]][241:388] %>% unlist()
file_validation_saiboat
file.copy(file_validation_saiboat, file.path(validation_sailboat_dir))
file_test_saiboat <- file_name[[10]][389:488] %>% unlist()
file_test_saiboat
file.copy(file_test_saiboat, file.path(test_sailboat_dir))


model_v1 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dropout(rate=0.3) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(rate=0.2) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 9, activation = "softmax")
model_v1 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(),
  metrics = c("acc")
)


datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40, # randomly rotate images up to 40 degrees
  width_shift_range = 0.2, # randomly shift 20% pictures horizontally
  height_shift_range = 0.2, # randomly shift 20% pictures vertically
  shear_range = 0.2, # randomly apply shearing transformations
  zoom_range = 0.2, # randomly zooming inside pictures
  horizontal_flip = TRUE, # randomly flipping half the images horizontally
  fill_mode = "nearest" # used for filling in newly created pixels
)



#train_datagen <- image_data_generator(rescale = 1/255)
#validation_datagen <- image_data_generator(rescale = 1/255)
#test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  "train/", # Target directory
  datagen, # Training data generator
  target_size = c(150, 150), # Resizes all images to 150 × 150
  batch_size = 20, # 20 samples in one batch
  class_mode = "categorical" # Because we use categorical_crossentropy loss,
  # we need categorical labels.
)

validation_generator <- flow_images_from_directory(
  "validation/",
  datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "categorical"
)

test_generator <- flow_images_from_directory(
  "test/",
  datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "categorical"
)




history_v1 <- model_v1 %>%
  fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 15,
    validation_data =
      validation_generator,
    validation_steps = 50
  )
plot(history_v1)

saveRDS(history_v1, "./model_history.rds")
save_model_hdf5(model_v1, "./final_model.h5")


