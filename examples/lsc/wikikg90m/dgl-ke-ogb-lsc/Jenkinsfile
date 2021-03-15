#!/usr/bin/env groovy

def init_git() {
  sh "rm -rf *"
  checkout scm
  sh "git submodule update --recursive --init"
}

def kg_test_linux(backend, dev) {
  init_git()
  timeout(time: 20, unit: 'MINUTES') {
    sh "bash tests/scripts/task_kg_test.sh ${backend} ${dev}"
  }
}

pipeline {
  agent any
  stages {
    stage("Lint Check") {
      agent { 
        docker {
          label "linux-cpu-node"
          image "dgllib/dgl-ci-lint" 
        }
      }
      steps {
        init_git()
        sh "bash tests/scripts/task_lint.sh"
      }
      post {
        always {
          cleanWs disableDeferredWipeout: true, deleteDirs: true
        }
      }
    }
    stage("App") {
      parallel {
        stage("Knowledge Graph CPU") {
          agent { 
            docker {
              label "linux-cpu-node"
              image "dgllib/dgl-ci-cpu:conda" 
            }
          }
          stages {
            stage("Torch test") {
              steps {
                kg_test_linux("pytorch", "cpu")
              }
            }
            stage("MXNet test") {
              steps {
                kg_test_linux("mxnet", "cpu")
              }
            }
          }
          post {
            always {
              cleanWs disableDeferredWipeout: true, deleteDirs: true
            }
          }
        }
        stage("Knowledge Graph GPU") {
          agent {
            docker {
              label "linux-gpu-node"
              image "dgllib/dgl-ci-gpu:conda"
              args "--runtime nvidia"
            }
          }
          stages {
            stage("Torch test") {
              steps {
                kg_test_linux("pytorch", "gpu")
              }
            }
            stage("MXNet test") {
              steps {
                kg_test_linux("mxnet", "gpu")
              }
            }
          }
          post {
            always {
              cleanWs disableDeferredWipeout: true, deleteDirs: true
            }
          }
        }
      }
    }
  }
}
