# Argo CD Demo Runbook

Инструкция для повторного запуска демонстрации после перезагрузки ноутбука.

## 1. Запустить Docker Desktop

Открой Docker Desktop и дождись, пока он полностью стартует.

## 2. Запустить minikube

В PowerShell или cmd:

```cmd
"C:\Program Files\Kubernetes\Minikube\minikube.exe" start
```

Проверка:

```cmd
kubectl get nodes
```

Ожидаемо:

```text
minikube   Ready
```

## 3. Проверить Argo CD

```cmd
kubectl -n argocd get pods
kubectl -n argocd get applications
```

Ожидаемо:

```text
aml-monitoring   Synced   Progressing
```

`Progressing` допустим: он может появляться из-за падающего drift CronJob. Главное для демонстрации Argo CD: `Synced`.

Если приложения нет, создать его заново:

```cmd
kubectl apply -f k8s/argocd/application.yaml
```

## 4. Проверить приложение

```cmd
kubectl -n aml-monitoring get pods
```

Основные pod должны быть `Running`:

```text
aml-backend
aml-frontend
aml-mlflow
grafana
prometheus
```

## 5. Открыть UI через port-forward

Открой три отдельных терминала.

Argo CD UI:

```cmd
kubectl -n argocd port-forward svc/argocd-server 8081:443
```

Frontend из Kubernetes:

```cmd
kubectl -n aml-monitoring port-forward svc/aml-frontend 18080:80
```

Backend из Kubernetes:

```cmd
kubectl -n aml-monitoring port-forward svc/aml-backend 18000:8000
```

Открыть в браузере:

```text
Argo CD: https://localhost:8081
Frontend: http://localhost:18080
Backend health: http://localhost:18000/health
```

Argo CD login:

```text
username: admin
password: NffuLj7tZIFAHBW6
```

Если пароль изменился после пересоздания Argo CD, получить новый в PowerShell:

```powershell
$p = kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}"
[System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($p))
```

## 6. Что показать преподавателю

Команды:

```cmd
kubectl -n argocd get applications
kubectl -n aml-monitoring get pods
```

В Argo CD UI показать приложение `aml-monitoring`:

- sync status: `Synced`;
- repository: `https://github.com/alukiko/mlops-aml-transactions.git`;
- target revision: `argocd-demo`;
- path: `k8s/base`;
- дерево ресурсов: `Deployment`, `Service`, `ConfigMap`, `CronJob`, `Job`.

## 7. Проверка self-heal

Для демонстрации GitOps self-heal лучше менять Deployment, а не удалять pod:

```cmd
kubectl -n aml-monitoring scale deployment aml-backend --replicas=0
```

Через несколько секунд проверить:

```cmd
kubectl -n aml-monitoring get deployment aml-backend
```

Argo CD должен вернуть `replicas=1`, потому что в Git-манифесте указано `replicas: 1`, а в Application включен `selfHeal: true`.
