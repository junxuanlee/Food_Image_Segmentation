from config import *
from unet_model import unet
from real_aug import image_generator

def generators():
    train_generator = image_generator(p1,p2,TARGET_SIZE,BATCH_SIZE,True,3.0,num_classes_to_keep,num_filest)
    val_generator = image_generator(p3,p4,TARGET_SIZE,BATCH_SIZE,False,3.0,num_classes_to_keep,num_filesv)
    return train_generator, val_generator

def view_transform(image):
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(image)

def dice_coef(y_true,y_pred,smooth=1):
    y_true = tf.one_hot(tf.cast(y_true,tf.int32),n_classes)
    
    y_pred = tf.argmax(y_pred, axis=-1)

    y_pred = tf.one_hot(tf.cast(y_pred,tf.int32),n_classes)
    
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f*y_pred_f)
    return (2. * intersection*smooth)/(tf.keras.backend.sum(y_true_f)+tf.keras.backend.sum(y_pred_f)+smooth)

def display(display_list,name):
    plt.figure(figsize=(10, 10))

    title = ['Input Image', 'True '+name+' Mask', 'Predicted '+name+' Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))

        plt.axis('off')
    plt.show()

def display_test(display_list):
    plt.figure(figsize=(10, 10))

    title = ['Input Image', 'True mask', 'Predicted mask','Filtered mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))

        plt.axis('off')
    plt.show()

def display_test_real_images(display_list):
    plt.figure(figsize=(10, 10))

    title = ['Input Image','Predicted mask','Filtered mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        #plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))

        plt.axis('off')
    plt.show()

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(epoch='test'):
    
    image = cv2.imread(image_path)
    image = cv2.resize(image,TARGET_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_for_model = np.expand_dims(image,axis=0)/255.0
    
    food_mask = cv2.imread(food_mask_path)
    food_mask = cv2.resize(food_mask,TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    food_mask = cv2.cvtColor(food_mask, cv2.COLOR_RGB2GRAY) 
    food_mask = np.expand_dims(food_mask,-1)
    
    if num_classes_to_keep:
        food_mask[food_mask>num_classes_to_keep] = 0.0

    pred_mask_food = unet.predict(image_for_model)

    pred_food_mask = pred_mask_food
    
    display([image, food_mask, create_mask(pred_food_mask)],'Food')

    
    path = training_progress_path
    
    title = ['Epoch '+str(epoch)+': Food']
    masks = [create_mask(pred_food_mask)]
    for i in range(1):
        plt.subplot(1, len(masks), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(masks[i]))

    plt.savefig(path+'epoch_'+str(epoch)+'.png')
    print('saved')

    #========================================================================
    image = cv2.imread(image_path2)
    image = cv2.resize(image,TARGET_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_for_model = np.expand_dims(image,axis=0)/255.0

    pred_mask_food = unet.predict(image_for_model)

    pred_food_mask = pred_mask_food
    
    display([image, create_mask(pred_food_mask)],'Food')

    path = training_progress_path
    
    title = ['Epoch '+str(epoch)+': Food']
    masks = [create_mask(pred_food_mask)]
    for i in range(1):
        plt.subplot(1, len(masks), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(masks[i]))

    plt.savefig(path+'epoch_'+str(epoch)+'_home.png')
    print('saved')

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        #clear_output(wait=True)
        show_predictions(epoch)
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

def scheduler(epoch, lr):
    if epoch < 35:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
