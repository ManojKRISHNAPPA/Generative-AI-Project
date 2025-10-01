#!/bin/bash

#k8s-deployment.sh

sed -i "s#replace#${imageName}#g" Deployment.yaml
# kubectl -n default get deployment ${deploymentName} > /dev/null

# if [[ $? -ne 0 ]]; then
#     echo "deployment ${deploymentName} doesnt exist"
#     kubectl -n default apply -f Deployment.yml
# else
#     echo "deployment ${deploymentName} exist"
#     echo "image name - ${imageName}"
#     kubectl -n default set image deploy ${deploymentName} ${containerName}=${imageName} --record=true
# fi


kubectl -n microdegree apply -f Deployment.yaml