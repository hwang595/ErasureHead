KEY_PEM_NAME=HongyiScript.pem
export DEEPLEARNING_WORKERS_COUNT=`wc -l < hosts`

sudo bash -c "cat hosts >> /etc/hosts"
cp config ~/.ssh/

cd ~/.ssh
eval `ssh-agent -s`
ssh-add ${KEY_PEM_NAME}
ssh-keygen -t rsa -b 4096 -C "hongyiwang.hdu@gmail.com"

for i in $(seq 1 $DEEPLEARNING_WORKERS_COUNT);
  do
  scp -i ${KEY_PEM_NAME} id_rsa.pub deeplearning-worker${i}:~/.ssh
  ssh -i ${KEY_PEM_NAME} deeplearning-worker${i} 'cd ~/.ssh; cat id_rsa.pub >> authorized_keys'
  scp -i ${KEY_PEM_NAME} -r ${HOME}/approximate_coding_gd deeplearning-worker${i}:~/
  #ssh -i ${KEY_PEM_NAME} deeplearning-worker${i} 'bash ${HOME}/approximate_coding_gd/data_prepare.sh'
  echo "Done writing public key to worker: deeplearning-worker${i}"
 done
