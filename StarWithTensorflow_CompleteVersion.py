import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
linear_model = W*x+b
loss = tf.reduce_sum(tf.square(linear_model-y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# train data
x_train=[1,2,3,4]
y_train=[0,-1,-2,-3]

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

for i in range(5000):
   sess.run(train,{x:x_train, y:y_train})
   
curr_W,curr_b,curr_loss = sess.run([W,b,loss],{x:x_train, y:y_train})
print ("W: %s b: %s loss: %s"%(curr_W,curr_b,curr_loss))



