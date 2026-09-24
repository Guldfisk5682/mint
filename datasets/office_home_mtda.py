import os.path as osp
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden


def _normalize_domain_name(domain_name):
    aliases = {
        "a": "art",
        "art": "art",
        "artistic": "art",
        "c": "clipart",
        "clipart": "clipart",
        "p": "product",
        "product": "product",
        "r": "real_world",
        "real": "real_world",
        "real_world": "real_world",
        "real-world": "real_world",
        "real world": "real_world",
        "real_worlds": "real_world",
    }
    key = domain_name.strip().lower().replace("-", "_")
    key = key.replace(" ", "_")
    if key not in aliases:
        raise ValueError(f"Unsupported Office-Home domain: {domain_name}")
    return aliases[key]


@DATASET_REGISTRY.register()
class OfficeHomeMTDA(DatasetBase):
    """Office-Home single-source multi-target domain adaptation protocol."""

    dataset_dir = "office_home"
    domains = ["art", "clipart", "product", "real_world"]
    domain_codes = OrderedDict(
        [
            ("art", "A"),
            ("clipart", "C"),
            ("product", "P"),
            ("real_world", "R"),
        ]
    )

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        if not osp.isdir(self.dataset_dir):
            raise FileNotFoundError(
                "OfficeHomeMTDA expects a dataset directory at "
                f'"{self.dataset_dir}". Please set DATA to the parent directory '
                "that contains office_home/."
            )

        source_domains = [_normalize_domain_name(d) for d in cfg.DATASET.SOURCE_DOMAINS]
        if len(source_domains) != 1:
            raise ValueError(
                "OfficeHomeMTDA requires exactly one source domain, "
                f"but got {source_domains}"
            )
        self.source_domain = source_domains[0]

        expected_targets = [d for d in self.domains if d != self.source_domain]
        raw_targets = list(getattr(cfg.DATASET, "TARGET_DOMAINS", []))
        if not raw_targets or raw_targets == ["auto"]:
            target_domains = expected_targets
        else:
            target_domains = [_normalize_domain_name(d) for d in raw_targets]

        if sorted(target_domains) != sorted(expected_targets):
            raise ValueError(
                "OfficeHomeMTDA expects the three remaining domains as targets. "
                f"Source={self.source_domain}, expected targets={expected_targets}, "
                f"but got {target_domains}"
            )

        self.target_domains = target_domains
        self.check_input_domains([self.source_domain], self.target_domains)

        class_to_label = self._build_label_mapping(self.source_domain)
        train_x = self._read_domain(self.source_domain, class_to_label)

        train_u_by_domain = OrderedDict()
        test_by_domain = OrderedDict()
        train_u = []
        test = []

        for domain_name in self.target_domains:
            items = self._read_domain(domain_name, class_to_label)
            train_u_by_domain[domain_name] = items
            test_by_domain[domain_name] = list(items)
            train_u.extend(items)
            test.extend(items)

        super().__init__(train_x=train_x, train_u=train_u, test=test)

        self.train_u_by_domain = train_u_by_domain
        self.test_by_domain = test_by_domain

    def _domain_dir(self, domain_name):
        domain_dir = osp.join(self.dataset_dir, domain_name)
        if not osp.isdir(domain_dir):
            raise FileNotFoundError(
                f'Missing Office-Home domain directory: "{domain_dir}". '
                "Please verify the dataset layout under DATA/office_home/."
            )
        return domain_dir

    def _build_label_mapping(self, reference_domain):
        domain_dir = self._domain_dir(reference_domain)
        class_names = listdir_nohidden(domain_dir)
        class_names.sort()
        return OrderedDict((class_name, label) for label, class_name in enumerate(class_names))

    def _read_domain(self, domain_name, class_to_label):
        items = []
        domain_dir = self._domain_dir(domain_name)
        class_names = listdir_nohidden(domain_dir)
        class_names.sort()

        expected_classes = list(class_to_label.keys())
        if class_names != expected_classes:
            raise ValueError(
                f"Closed-set Office-Home expects shared classes across domains. "
                f"Domain {domain_name} has classes {class_names[:5]}..., "
                f"expected {expected_classes[:5]}..."
            )

        domain_index = self.domains.index(domain_name)
        for class_name in class_names:
            class_path = osp.join(domain_dir, class_name)
            imnames = listdir_nohidden(class_path)
            for imname in imnames:
                impath = osp.join(class_path, imname)
                items.append(
                    Datum(
                        impath=impath,
                        label=class_to_label[class_name],
                        domain=domain_index,
                        classname=class_name.lower(),
                    )
                )

        return items
