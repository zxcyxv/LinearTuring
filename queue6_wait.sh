cd /workspace/mnist_model1
while ! grep -q "QUEUE5 DONE" runs/queue5.log; do sleep 20; done
bash queue6.sh
