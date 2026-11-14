pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                echo '1. Pulling code from GitHub...'
                checkout scm
            }
        }

        stage('Setup Environment & Dependencies') {
            steps {
                echo '2. Setup Python venv...'
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements-test.txt
                '''
            }
        }

        stage('Syntax / Linting Check') {
            steps {
                echo '3. Syntax checking...'
                sh '''
                    . venv/bin/activate
                    python3 -m py_compile *.py || true
                '''
            }
        }

        stage('Execute Automated Tests') {
            steps {
                echo '4. Running Pytest...'
                sh '''
                    . venv/bin/activate
                    mkdir -p test-reports
                    pytest --junitxml=test-reports/results.xml test_model_logic.py
                '''
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'test-reports/*.xml'
            cleanWs deleteDirs: true, notFailBuild: true
        }
    }
}
