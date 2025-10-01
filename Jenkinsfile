pipeline {
    agent any

    environment {
        AWS_REGION = "us-east-1"
        CLUSTER_NAME = "microdegree"
        NAMESPACE = "microdegree"
        DEPLOYMENT_NAME = "openai-chatbot"
        SERVICE_NAME = "openai-chatbot-service"
        IMAGE_NAME = "manojkrishnappa/genai-openai:${GIT_COMMIT}"
    }

    stages {
        stage('Git Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/ManojKRISHNAPPA/Generative-AI-Project.git'
            }
        }

        stage('Build & Tag Docker Image') {
            steps {
                script {
                    sh 'docker build -t manojkrishnappa/genai-openai:${GIT_COMMIT} .'
                }
            }
        }

        stage('Docker Image Scan') {
            steps {
                script {
                    sh 'trivy image --format table -o trivy-image-report.html manojkrishnappa/genai-openai:${GIT_COMMIT}'
                }
            }
        }

        stage('Login to Docker Hub') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                    sh "echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin"
                }
            }
        }

        stage('Push Docker Image') {
            steps {
                sh 'docker push manojkrishnappa/genai-openai:${GIT_COMMIT}'
            }
        }

        stage('Python SonarQube Analysis') {
            steps {
                sh """
                    pip install --upgrade pip pytest coverage
                    coverage run -m pytest
                    coverage xml -o coverage.xml

                    sonar-scanner \
                      -Dsonar.projectKey=OpenAI \
                      -Dsonar.sources=. \
                      -Dsonar.host.url=http://3.85.22.155:9000 \
                      -Dsonar.login=daada275b6ba2babdd26e784a8133bd4fea10379 \
                      -Dsonar.python.coverage.reportPaths=coverage.xml
                """
            }
        }

        stage('Quality Gate') {
            steps {
                timeout(time: 2, unit: 'MINUTES') {
                    waitForQualityGate abortPipeline: true
                }
            }
        }

        stage('Update EKS Config') {
            steps {
                sh "aws eks update-kubeconfig --region ${AWS_REGION} --name ${CLUSTER_NAME}"
            }
        }

        stage('Deploy to EKS') {
            steps {
                withKubeConfig(
                    caCertificate: '',
                    clusterName: 'microdegree',
                    contextName: '',
                    credentialsId: 'kube',
                    namespace: "${NAMESPACE}",
                    restrictKubeConfigAccess: false,
                    serverUrl: 'https://E8A1E7C9DE8BC222DA09253C5200F1E3.gr7.us-east-1.eks.amazonaws.com'
                ) {
                    sh "sed -i 's|replace|${IMAGE_NAME}|g' Deployment.yaml"
                    sh "kubectl apply -f Deployment.yaml -n ${NAMESPACE}"
                    sh "kubectl rollout status deployment/${DEPLOYMENT_NAME} -n ${NAMESPACE}"
                }
            }
        }

        stage('Verify Deployment') {
            steps {
                withKubeConfig(
                    caCertificate: '',
                    clusterName: 'microdegree',
                    contextName: '',
                    credentialsId: 'kube',
                    namespace: "${NAMESPACE}",
                    restrictKubeConfigAccess: false,
                    serverUrl: 'https://E8A1E7C9DE8BC222DA09253C5200F1E3.gr7.us-east-1.eks.amazonaws.com'
                ) {
                    sh "kubectl get deployment -n ${NAMESPACE}"
                    sh "kubectl get pods -n ${NAMESPACE}"
                    sh "kubectl get svc -n ${NAMESPACE}"
                }
            }
        }
    }

    post {
        always {
            script {
                def jobName = env.JOB_NAME
                def buildNumber = env.BUILD_NUMBER
                def pipelineStatus = currentBuild.result ?: 'SUCCESS'
                def bannerColor = pipelineStatus.toUpperCase() == 'SUCCESS' ? 'green' : 'red'

                def body = """
                    <html>
                    <body>
                    <div style="border: 4px solid ${bannerColor}; padding: 10px;">
                        <h2>${jobName} - Build ${buildNumber}</h2>
                        <div style="background-color: ${bannerColor}; padding: 10px;">
                            <h3 style="color: white;">Pipeline Status: ${pipelineStatus.toUpperCase()}</h3>
                        </div>
                        <p>Check the <a href="${BUILD_URL}">console output</a>.</p>
                    </div>
                    </body>
                    </html>
                """

                emailext (
                    subject: "${jobName} - Build ${buildNumber} - ${pipelineStatus.toUpperCase()}",
                    body: body,
                    to: 'rohitpatil.cse@gmail.com,manojdevopstest@gmail.com',
                    from: 'manojdevopstest@gmail.com',
                    replyTo: 'manojdevopstest@gmail.com',
                    mimeType: 'text/html',
                    attachmentsPattern: 'trivy-image-report.html'
                )
            }
        }
    }
}
