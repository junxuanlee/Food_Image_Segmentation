def nutrition__calculator (food, volume):
    nutritions = {'bread'          :{'density':0.58, 'unit':32 , 'carbohydrate':12, 'protein':2   , 'fats':75  , 'fibre':1  , 'calorie':65},
                  'pasta'          :{'density':0.52, 'unit':140, 'carbohydrate':12, 'protein':8.1 , 'fats':1.3 , 'fibre':2.5, 'calorie':221},
                  'chicken'        :{'density':0.87, 'unit':85 , 'carbohydrate':12, 'protein':26  , 'fats':2.7 , 'fibre':0  , 'calorie':128},
                  'carrots'        :{'density':0.46, 'unit':61 , 'carbohydrate':12, 'protein':0.6 , 'fats':0   , 'fibre':1.7, 'calorie':25},
                  'pizza'          :{'density':1.07, 'unit':107, 'carbohydrate':12, 'protein':12.2, 'fats':10.4, 'fibre':2.5, 'calorie':285},
                  'cutlet'         :{'density':0.87, 'unit':85 , 'carbohydrate':12, 'protein':26  , 'fats':2.7 , 'fibre':0  , 'calorie':128},
                  'tangerines'     :{'density':1.05, 'unit':76 , 'carbohydrate':12, 'protein':0.6 , 'fats':0.2 , 'fibre':1.3, 'calorie':40},
                  'mashed_potatoes':{'density':0.97, 'unit':173, 'carbohydrate':12, 'protein':4.6 , 'fats':0.2 , 'fibre':4  , 'calorie':164},
                  'green_beans'    :{'density':1.01, 'unit':100, 'carbohydrate':12, 'protein':1.8 , 'fats':0.2 , 'fibre':2.7, 'calorie':31},
                  'spinach'        :{'density':0.80, 'unit':85 , 'carbohydrate':12, 'protein':2   , 'fats':0   , 'fibre':0  , 'calorie':20}}
    
    density               = nutritions[food]['density']
    unit                  = nutritions[food]['unit']
    carbohydrate_per_gram = nutritions[food]['carbohydrate'] / unit
    protein_per_gram      = nutritions[food]['protein'] / unit
    fats_per_gram         = nutritions[food]['fats'] / unit
    fibre_per_gram        = nutritions[food]['fibre'] / unit
    calorie_per_gram      = nutritions[food]['calorie'] / unit
                  
    weight = density*volume
    
    total_carbohydrate = weight*carbohydrate_per_gram
    total_protein      = weight*protein_per_gram
    total_fats         = weight*fats_per_gram
    total_fibre        = weight*fibre_per_gram
    total_calorie      = weight*calorie_per_gram

    
    return total_carbohydrate, total_protein, total_fats, total_fibre, total_calorie
        
    
