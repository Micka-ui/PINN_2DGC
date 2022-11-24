# This is a sample Python script.
import os.path
import time
from modelSave import *
from PhysicsInformedNN import *
import numpy as np


if __name__ == '__main__':
    H = 0.25
    L = 6 * H
    rho_max = 1002.4
    delta_rho = 0.20
    g = 9.81
    mu = 1e-3
    ub = (g * delta_rho * H / rho_max) ** 0.5
    Re = H * rho_max * ub / (mu)
    K = 1e-6
    nu = mu / rho_max
    Sc = nu / K
    ReSc = Re*Sc
    print(f"la vitesse moyenne est {ub}, le Reynolds vaut {Re}")
    print(f"le schmidt {Sc}, devant cons de la masse {1 / ReSc}")


    data = np.load('data/data_2D_Nek5000.npy')

    X_star,Y_star,T_star,Rho_star,U_star,V_star,P_star = np.array_split(data,7,axis=-1)

    ##Input data
    x_data = X_star.flatten().reshape(-1,1)
    y_data = Y_star.flatten().reshape(-1, 1)
    t_data = T_star.flatten().reshape(-1, 1)

    print('x adim : (%.3f,%.3f), pixel x : %s'%(x_data.min(),x_data.max(),X_star.shape[1]))
    print('y adim : (%.3f,%.3f), pixel y : %s' % (y_data.min(), y_data.max(),X_star.shape[0]))
    print('t adim : (%.3f,%.3f), timestep : %s' % (t_data.min(), t_data.max(),X_star.shape[2]))

    x_train = np.concatenate([x_data,y_data,t_data],axis=1)


    ##Observation data

    rho_data = Rho_star.flatten().reshape(-1, 1)
    u_data = U_star.flatten().reshape(-1, 1)
    v_data = V_star.flatten().reshape(-1, 1)


    y_train = np.concatenate([rho_data,rho_data*0.0],axis=-1)

    x_valid = np.copy(x_train)
    y_valid = np.concatenate([rho_data,u_data,v_data],axis=-1)



    path_log = './models'
    dirname = time.strftime('Run_%m_%M')
    name_dir = os.path.join(path_log,dirname)
    if not os.path.exists(name_dir):
        os.makedirs(name_dir)
    pt_log = os.path.join(name_dir,'loss.log')

    ###Callbacks for saving log
    csv_logger = tf.keras.callbacks.CSVLogger(pt_log,separator=',',append = True)

    ##Callback to save model each  frequence epoch
    frequence = 2
    call_model = modelSave(frequence,name_dir)

    ###PARAMETERS MODEL

    ##layers size
    num_layers = 7
    ##hidden_units
    hidden_units = [125 for i in range(num_layers)]
    x_test = x_train[:2,:]

    EPOCHS = 10
    BATCH_SIZE = 4096

    ##LR_SCHEDULER
    lr_start = 1e-3
    lr_end = 1e-5
    print('Training starts at Learning_rate :%s and ends at Learning_rate : %s'%(lr_start,lr_end))
    Cst = (lr_end/lr_start)**(1/EPOCHS)
    funct = lambda epoch : lr_start*(Cst**epoch)
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(funct)

    ##All callbacks
    callbacks = [csv_logger,call_model,lr_schedule]

    ##Loss and optimizer
    loss_fn = tf.keras.losses.MeanAbsoluteError()
    optimizer = tf.keras.optimizers.Adam()
    ##training metrics
    loss_rho = tf.keras.metrics.MeanAbsoluteError(name='loss_rho')
    loss_e1 = tf.keras.metrics.MeanSquaredError(name='loss_e1')
    loss_e2 = tf.keras.metrics.MeanSquaredError(name='loss_e2')
    loss_e3 = tf.keras.metrics.MeanSquaredError(name='loss_e3')
    loss_e4 = tf.keras.metrics.MeanSquaredError(name='loss_e4')
    training_metrics =[loss_rho,loss_e1,loss_e2,loss_e3,loss_e4]

    ##evaluation_metrics
    loss_rho_valid = tf.keras.metrics.MeanAbsoluteError(name='loss_rho')
    loss_u_valid = tf.keras.metrics.MeanAbsoluteError(name='loss_u')
    loss_v_valid = tf.keras.metrics.MeanAbsoluteError(name='loss_v')
    evaluation_metrics = [loss_rho_valid,loss_u_valid,loss_v_valid]

    model = PhysicsInformedNN(Re,Sc,hidden_units=hidden_units,\
                              x_test = x_test,\
                              loss_fn = loss_fn,\
                              training_metrics=training_metrics,\
                              evaluation_metrics=evaluation_metrics)
    model.compile(loss = loss_fn,optimizer = optimizer)




    model.fit(x = x_train , y = y_train, epochs= EPOCHS, batch_size= BATCH_SIZE,validation_data= (x_valid,y_valid),callbacks=callbacks)









