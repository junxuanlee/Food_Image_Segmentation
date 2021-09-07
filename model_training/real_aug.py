import numpy as np
import os
import cv2
import random
from random import randint

from matplotlib import pyplot as plt
import albumentations as A
import numpy as np
import tensorflow as tf

def view_transform(image):
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(image)

def create_augmentation_pipeline():

    oneOf0 = [A.FancyPCA(alpha=0.45),A.RGBShift (r_shift_limit=30, g_shift_limit=30, b_shift_limit=30)]
    oneOf1 = [A.RandomBrightness(limit=0.2),A.GaussNoise(),A.RandomGamma(),A.CLAHE()]#,A.CLAHE(),A.FancyPCA(alpha=0.4),A.RandomSunFlare()]
    oneOf2 = [A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45)] #A.ElasticTransform(alpha = 50,sigma = 50 * 0.05,alpha_affine = 50 * 0.03)
    oneOf3 = [A.CoarseDropout(max_holes=5, max_height=60, max_width=60,mask_fill_value=0)]#,A.RandomCrop(150, 150),A.CoarseDropout(max_holes=5, max_height=60, max_width=60,mask_fill_value=0)
    oneOf4 = [A.GridDistortion()]
    
    aug_block = [A.RandomRotate90(p=0.6), A.OneOf(oneOf0,p = 1), A.OneOf(oneOf1,p = 0.7), A.OneOf(oneOf2,p = 0.8),A.OneOf(oneOf3,p = 0.4),A.OneOf(oneOf4,p=0.5)]
    
    augmentation_pipeline = A.Compose(aug_block, p = 1)
    
    return augmentation_pipeline

#============================================================================================
# up&down
#============================================================================================
def updown(lst1,lst2,lst3):
    image1,image2,food_mask1,food_mask2,weight1,weight2 = lst1[0],lst1[1],lst2[0],lst2[1],lst3[0],lst3[1]
    
    #image
    half1 = image1[:64]
    half2 = image2[64:]
    new_img_half = np.concatenate([half1,half2])
    new_img_half_for_model = np.expand_dims(new_img_half,axis=0)/255.0

    #mask
    half1m = food_mask1[:64]
    half2m = food_mask2[64:]
    new_img_halfm = np.concatenate([half1m,half2m])
    new_img_halfm = np.expand_dims(new_img_halfm,-1)
    new_img_halfm = np.expand_dims(new_img_halfm,axis=0)

    #weight
    half1w = weight1[:64]
    half2w = weight2[64:]
    new_img_halfw = np.concatenate([half1w,half2w])
    new_img_halfw = np.expand_dims(new_img_halfw,-1)
    new_img_halfw = np.expand_dims(new_img_halfw,axis=0)
    
    return new_img_half_for_model,new_img_halfm,new_img_halfw

#============================================================================================
# left&right
#============================================================================================
def leftright(lst1,lst2,lst3):
    image1,image2,food_mask1,food_mask2,weight1,weight2 = lst1[0],lst1[1],lst2[0],lst2[1],lst3[0],lst3[1]
    #image
    half1 = image1[:128,:64]
    half2 = image2[:128,64:]
    new_img_half = np.concatenate([half1,half2],1)
    new_img_half_for_model = np.expand_dims(new_img_half,axis=0)/255.0

    #mask
    half1m = food_mask1[:128,:64]
    half2m = food_mask2[:128,64:]
    new_img_halfm = np.concatenate([half1m,half2m],1)
    new_img_halfm = np.expand_dims(new_img_halfm,-1)
    new_img_halfm = np.expand_dims(new_img_halfm,axis=0)

    #weight
    half1w = weight1[:128,:64]
    half2w = weight2[:128,64:]
    new_img_halfw = np.concatenate([half1w,half2w],1)
    new_img_halfw = np.expand_dims(new_img_halfw,-1)
    new_img_halfw = np.expand_dims(new_img_halfw,axis=0)
    
    return new_img_half_for_model,new_img_halfm,new_img_halfw

#============================================================================================
# quarter
#============================================================================================
def quarter(lst1,lst2,lst3):
    image1,image2,image3,image4,food_mask1,food_mask2,food_mask3,food_mask4,weight1,weight2,weight3,weight4 = lst1[0],lst1[1],lst1[2],lst1[3],lst2[0],lst2[1],lst2[2],lst2[3],lst3[0],lst3[1],lst3[2],lst3[3]
    #image
    quad1 = image1[:64,:64]
    quad2 = image2[:64,64:]
    quad3 = image3[64:,:64]
    quad4 = image4[64:,64:]
    onehalf1 = np.concatenate([quad1,quad2],1)
    onehalf2 = np.concatenate([quad3,quad4],1)
    full = np.concatenate([onehalf1,onehalf2])
    full_for_model = np.expand_dims(full,axis=0)/255.0

    #mask
    quad1m = food_mask1[:64,:64]
    quad2m = food_mask2[:64,64:]
    quad3m = food_mask3[64:,:64]
    quad4m = food_mask4[64:,64:]
    onehalf1m = np.concatenate([quad1m,quad2m],1)
    onehalf2m = np.concatenate([quad3m,quad4m],1)
    fullm = np.concatenate([onehalf1m,onehalf2m])
    fullm = np.expand_dims(fullm,-1)
    fullm = np.expand_dims(fullm,axis=0)

    #weight
    quad1w = weight1[:64,:64]
    quad2w = weight2[:64,64:]
    quad3w = weight3[64:,:64]
    quad4w = weight4[64:,64:]
    onehalf1w = np.concatenate([quad1w,quad2w],1)
    onehalf2w = np.concatenate([quad3w,quad4w],1)
    fullw = np.concatenate([onehalf1w,onehalf2w])
    fullw = np.expand_dims(fullw,-1)
    fullw = np.expand_dims(fullw,axis=0)

    return full_for_model, fullm, fullw

#============================================================================================
# 4 vertical bars
#============================================================================================
def vertical4bars(lst1,lst2,lst3):
    image1,image2,image3,image4,food_mask1,food_mask2,food_mask3,food_mask4,weight1,weight2,weight3,weight4 = lst1[0],lst1[1],lst1[2],lst1[3],lst2[0],lst2[1],lst2[2],lst2[3],lst3[0],lst3[1],lst3[2],lst3[3]
    #image
    bar1 = image1[:128,:32]
    bar2 = image2[:128,32:64]
    bar3 = image3[:128,64:96]
    bar4 = image4[:128,96:128]
    first_half = np.concatenate([bar1,bar2],1)
    second_half = np.concatenate([bar3,bar4],1)
    full = np.concatenate([first_half,second_half],1)
    full_for_model = np.expand_dims(full,axis=0)/255.0

    #mask
    bar1m = food_mask1[:128,:32]
    bar2m = food_mask2[:128,32:64]
    bar3m = food_mask3[:128,64:96]
    bar4m = food_mask4[:128,96:128]
    first_halfm = np.concatenate([bar1m,bar2m],1)
    second_halfm = np.concatenate([bar3m,bar4m],1)
    fullm = np.concatenate([first_halfm,second_halfm],1)
    fullm = np.expand_dims(fullm,-1)
    fullm = np.expand_dims(fullm,axis=0)

    #weight
    quad1w = weight1[:128,:32]
    quad2w = weight2[:128,32:64]
    quad3w = weight3[:128,64:96]
    quad4w = weight4[:128,96:128]
    onehalf1w = np.concatenate([quad1w,quad2w],1)
    onehalf2w = np.concatenate([quad3w,quad4w],1)
    fullw = np.concatenate([onehalf1w,onehalf2w],1)
    fullw = np.expand_dims(fullw,-1)
    fullw = np.expand_dims(fullw,axis=0)
    
    return full_for_model, fullm, fullw

#============================================================================================
# 4 horizontal bars
#============================================================================================
def horizontal4bars(lst1,lst2,lst3):
    image1,image2,image3,image4,food_mask1,food_mask2,food_mask3,food_mask4,weight1,weight2,weight3,weight4 = lst1[0],lst1[1],lst1[2],lst1[3],lst2[0],lst2[1],lst2[2],lst2[3],lst3[0],lst3[1],lst3[2],lst3[3]
    
    #image
    bar1 = image1[:32,:128]
    bar2 = image2[32:64,:128]
    bar3 = image3[64:96,:128]
    bar4 = image4[96:128,:128]
    first_half = np.concatenate([bar1,bar2])
    second_half = np.concatenate([bar3,bar4])
    full = np.concatenate([first_half,second_half])
    full_for_model = np.expand_dims(full,axis=0)/255.0

    #mask
    bar1m = food_mask1[:32,:128]
    bar2m = food_mask2[32:64,:128]
    bar3m = food_mask3[64:96,:128]
    bar4m = food_mask4[96:128,:128]
    first_halfm = np.concatenate([bar1m,bar2m])
    second_halfm = np.concatenate([bar3m,bar4m])
    fullm = np.concatenate([first_halfm,second_halfm])
    fullm = np.expand_dims(fullm,-1)
    fullm = np.expand_dims(fullm,axis=0)

    #weight
    quad1w = weight1[:32,:128]
    quad2w = weight2[32:64,:128]
    quad3w = weight3[64:96,:128]
    quad4w = weight4[96:128,:128]
    onehalf1w = np.concatenate([quad1w,quad2w])
    onehalf2w = np.concatenate([quad3w,quad4w])
    fullw = np.concatenate([onehalf1w,onehalf2w])
    fullw = np.expand_dims(fullw,-1)
    fullw = np.expand_dims(fullw,axis=0)
    
    return full_for_model, fullm,fullw

#============================================================================================
# 16 boxes from 3 images
#============================================================================================
def merge3img(lst1,lst2,lst3):
    image1,image2,image3,food_mask1,food_mask2,food_mask3,weight1,weight2,weight3 = lst1[0],lst1[1],lst1[2],lst2[0],lst2[1],lst2[2],lst3[0],lst3[1],lst3[2]
        
    #image
    box1 = image1[:32,:32]
    box2 = image1[32:64,32:64]
    box3 = image1[64:96,64:96]
    box4 = image1[96:128,96:128]
    box5 = image2[96:128,:32]
    box6 = image2[64:96,32:64]
    box7 = image2[32:64,64:96]
    box8 = image2[:32,96:128]

    box9 = image3[32:64,:32]
    box10 = image3[64:96,:32]
    box11 = image3[:32,32:64]
    box12 = image3[:32,64:96]
    box13 = image3[32:64,96:128]
    box14 = image3[64:96,96:128]
    box15 = image3[96:128,32:64]
    box16 = image3[96:128,64:96]

    con1 = np.concatenate([box1,box9,box10,box5])
    con2 = np.concatenate([box11,box2,box6,box16])
    con3 = np.concatenate([box12,box7,box3,box15])
    con4 = np.concatenate([box8,box13,box14,box4])

    full = np.concatenate([con1,con2,con3,con4],1)
    full_for_model = np.expand_dims(full,axis=0)/255.0

    #mask
    box1m = food_mask1[:32,:32]
    box2m = food_mask1[32:64,32:64]
    box3m = food_mask1[64:96,64:96]
    box4m = food_mask1[96:128,96:128]
    box5m = food_mask2[96:128,:32]
    box6m = food_mask2[64:96,32:64]
    box7m = food_mask2[32:64,64:96]
    box8m = food_mask2[:32,96:128]

    box9m = food_mask3[32:64,:32]
    box10m = food_mask3[64:96,:32]
    box11m = food_mask3[:32,32:64]
    box12m = food_mask3[:32,64:96]
    box13m = food_mask3[32:64,96:128]
    box14m = food_mask3[64:96,96:128]
    box15m = food_mask3[96:128,32:64]
    box16m = food_mask3[96:128,64:96]

    con1m = np.concatenate([box1m,box9m,box10m,box5m])
    con2m = np.concatenate([box11m,box2m,box6m,box16m])
    con3m = np.concatenate([box12m,box7m,box3m,box15m])
    con4m = np.concatenate([box8m,box13m,box14m,box4m])

    fullm = np.concatenate([con1m,con2m,con3m,con4m],1)
    fullm = np.expand_dims(fullm,-1)
    fullm = np.expand_dims(fullm,axis=0)

    #weight
    box1w = weight1[:32,:32]
    box2w = weight1[32:64,32:64]
    box3w = weight1[64:96,64:96]
    box4w = weight1[96:128,96:128]
    box5w = weight2[96:128,:32]
    box6w = weight2[64:96,32:64]
    box7w = weight2[32:64,64:96]
    box8w = weight2[:32,96:128]

    box9w = weight3[32:64,:32]
    box10w = weight3[64:96,:32]
    box11w = weight3[:32,32:64]
    box12w = weight3[:32,64:96]
    box13w = weight3[32:64,96:128]
    box14w = weight3[64:96,96:128]
    box15w = weight3[96:128,32:64]
    box16w = weight3[96:128,64:96]

    con1w = np.concatenate([box1w,box9w,box10w,box5w])
    con2w = np.concatenate([box11w,box2w,box6w,box16w])
    con3w = np.concatenate([box12w,box7w,box3w,box15w])
    con4w = np.concatenate([box8w,box13w,box14w,box4w])

    fullw = np.concatenate([con1w,con2w,con3w,con4w],1)
    fullw = np.expand_dims(fullw,-1)
    fullw = np.expand_dims(fullw,axis=0)
    
    return full_for_model, fullm, fullw

def num():
    k=[1]
    while True:
        random.shuffle(k)
        for i in k:
            yield i

def nim():
    k=[0,1,2,3,4,5]
    while True:
        random.shuffle(k)
        for i in k:
            yield i

def image_generator(p1,p2,TARGET_SIZE,BATCH_SIZE,training,w,no,num_files):

    files1 = os.listdir(p1)
    files2 = os.listdir(p2)

    augment = False
    crop = False
    nom=1
    switch = num()
    crop_mode = nim()

    epoch=0
    while True:
        r = randint(0, 100000)
        random.Random(r).shuffle(files1)
        random.Random(r).shuffle(files2)

        for i in range(len(files1)):
            if training:
                if nom==1:
                    augment = False
                    crop = False
                if nom==2:
                    augment = True
                    crop = False
                if nom==3:
                    augment = False
                    crop = True

            if i%BATCH_SIZE == 0:
                image_con = cv2.imread(p1+files1[i])
                image_con = cv2.resize(image_con,TARGET_SIZE)
                image_con = cv2.cvtColor(image_con, cv2.COLOR_BGR2RGB)
                #image_con = np.expand_dims(image_con,axis=0)

                mask_con = cv2.imread(p2+files2[i])
                mask_con = cv2.resize(mask_con,TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                mask_con = np.expand_dims(mask_con[:,:,0],axis=-1)
                #mask_con = np.expand_dims(mask_con,axis=0)

                #classes = ['chips','banana','carrots','fish','peach','potatoes','broccoli','sandwich','custard','mashed_potatoes','sausage_roll',
                #           'cheese_pizza','bread','chicken','minted_summer_vegetables','cake','toast','cottage_pie','peas','jam_sponge',
                #           'biscuits','rice','mashed_swede','orange','fortified_tomato_&_lentil_soup','chicken_&_ham_pie','bbq_pork_&_pepper',
                #           'apple','pasta','vegetable_soup']

                #values = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

                if no:
                    mask_con[mask_con>no] = 0.0

                weight_shape = mask_con.shape
                sample_weight_con = np.ones(shape=weight_shape)
                sample_weight_con[mask_con != 0.0] = w

                if augment:
                    aug_pipeline = create_augmentation_pipeline()
                    augmented = aug_pipeline(image=image_con, masks=[mask_con,sample_weight_con])
                    image_con, mask_con, sample_weight_con = augmented['image'], augmented['masks'][0], augmented['masks'][1]

                    sample_weight_con[sample_weight_con==0.0] = 1.0

                image_con = np.expand_dims(image_con,axis=0)/255.0
                mask_con = np.expand_dims(mask_con,axis=0)
                sample_weight_con = np.expand_dims(sample_weight_con,axis=0)

            else:
                if crop: #crop
                    #ft = randint(0,5)
                    ft = next(crop_mode)
                    functions = [updown,leftright,quarter,vertical4bars,horizontal4bars,merge3img]
                    num_img = [2,2,4,4,4,3]

                    img = []
                    msk = []
                    wgt = []
                    for k in range(num_img[ft]):
                        idx = i+k
                        if idx > (len(files1)-1):
                            # 202 - (200-1)
                            idx = (idx - (len(files1)-1))-1
                        
                        #r = randint(0,num_files-1)
                        image = cv2.imread(p1+files1[idx])
                        image = cv2.resize(image,TARGET_SIZE)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        mask = cv2.imread(p2+files2[idx])
                        mask = cv2.resize(mask,TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

                        if no:
                            mask[mask>no] = 0.0

                        weight_shape = mask.shape
                        sample_weight = np.ones(shape=weight_shape)
                        sample_weight[mask != 0.0] = w      

                        img.append(image)
                        msk.append(mask)
                        wgt.append(sample_weight)

                    image,mask,sample_weight = functions[ft](img,msk,wgt)

                    image = image/255.0

                    image_con = np.concatenate([image_con,image],axis=0)
                    mask_con = np.concatenate([mask_con,mask],axis=0)
                    sample_weight_con = np.concatenate([sample_weight_con,sample_weight],axis=0)
            

                else: #augment
                    image = cv2.imread(p1+files1[i])
                    image = cv2.resize(image,TARGET_SIZE)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    #image = np.expand_dims(image,axis=0)

                    mask = cv2.imread(p2+files2[i])
                    mask = cv2.resize(mask,TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                    mask = np.expand_dims(mask[:,:,0],axis=-1)
                    #mask = np.expand_dims(mask,axis=0) 

                    if no:
                        mask[mask>no] = 0.0

                    weight_shape = mask.shape
                    sample_weight = np.ones(shape=weight_shape)
                    sample_weight[mask != 0.0] = w

                    if augment:
                        aug_pipeline = create_augmentation_pipeline()
                        augmented = aug_pipeline(image=image, masks=[mask,sample_weight])
                        image, mask, sample_weight = augmented['image'], augmented['masks'][0], augmented['masks'][1]

                        sample_weight[sample_weight==0.0] = 1.0

                    image = np.expand_dims(image,axis=0)/255.0
                    mask = np.expand_dims(mask,axis=0)
                    sample_weight = np.expand_dims(sample_weight,axis=0)

                    image_con = np.concatenate([image_con,image],axis=0)
                    mask_con = np.concatenate([mask_con,mask],axis=0)
                    sample_weight_con = np.concatenate([sample_weight_con,sample_weight],axis=0)


            if i !=0 and i%(BATCH_SIZE) == BATCH_SIZE-1:
                yield ((image_con,mask_con,sample_weight_con))

        if epoch%1 ==0:
            nom = next(switch)
        epoch=epoch+1
        
       
            


#p11 = 'C:/Users/Admin/Desktop/lol/train/images/'
#p22 = 'C:/Users/Admin/Desktop/lol/mask/images/'

#p1 = 'C:/Users/Admin/Desktop/jun/University/Year 4/GDP/work/code/notebooks/run/dataset/UNIMIB2016 Food Database/train/food_images/images/'
#p2 = 'C:/Users/Admin/Desktop/jun/University/Year 4/GDP/work/code/notebooks/run/dataset/UNIMIB2016 Food Database/train/food_masks/images/'

#TARGET_SIZE = (128,128)
#BATCH_SIZE = 6

#q = image_generator(p1,p2,TARGET_SIZE,BATCH_SIZE,False,3.0,0,num_files)
#a = next(q)
#b = next(q)
#c = next(q)
#d = next(q)

#figure1 = tf.keras.preprocessing.image.array_to_img(b[0][3])
#view_transform(figure1)

#figure1 = tf.keras.preprocessing.image.array_to_img(b[1][3])
#view_transform(figure1)

#figure1 = tf.keras.preprocessing.image.array_to_img(b[2][3])
#view_transform(figure1)























