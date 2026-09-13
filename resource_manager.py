"""
Resource manager.

The only module that talks to the Kubernetes API directly. Two jobs:
reads the deployment's ACTUAL current replica count (the reactive HPA
calculator needs ground truth, not "whatever we last commanded" -- those
can diverge from a manual kubectl scale, a failed rollout, etc.), and
applies a FinalDecision by patching the scale subresource.
"""

import time
from dataclasses import dataclass
from typing import Optional

from kubernetes import client, config
from kubernetes.client.rest import ApiException

from hpa_calculator import ScaleAction
from safety import FinalDecision


@dataclass
class ResourceManagerConfig:
    namespace: str = "default"
    deployment_name: str = "app-deployment"
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0


@dataclass
class ScaleResult:
    success: bool
    replicas_applied: Optional[int]
    error: Optional[str] = None


def _load_kube_client() -> client.AppsV1Api:
    """In-cluster config when running as a pod, falling back to the local
    kubeconfig for dev-machine runs — same code path either way."""
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client.AppsV1Api()


class ResourceManager:
    def __init__(self, cfg: ResourceManagerConfig = ResourceManagerConfig(), api_client=None):
        self.cfg = cfg
        # api_client injection point lets tests pass a mock instead of a real one
        self._api = api_client if api_client is not None else _load_kube_client()

    def get_current_replicas(self) -> int:
        scale = self._api.read_namespaced_deployment_scale(
            name=self.cfg.deployment_name, namespace=self.cfg.namespace
        )
        return scale.spec.replicas

    def apply(self, decision: FinalDecision) -> ScaleResult:
        if decision.action == ScaleAction.MAINTAIN:
            # no-op: don't generate a patch call (or a rollout event) for
            # a replica count that isn't changing
            return ScaleResult(success=True, replicas_applied=decision.replicas)

        body = {"spec": {"replicas": decision.replicas}}
        last_error = None

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                self._api.patch_namespaced_deployment_scale(
                    name=self.cfg.deployment_name,
                    namespace=self.cfg.namespace,
                    body=body,
                )
                return ScaleResult(success=True, replicas_applied=decision.replicas)
            except ApiException as e:
                last_error = str(e)
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_backoff_seconds * attempt)

        return ScaleResult(success=False, replicas_applied=None, error=last_error)