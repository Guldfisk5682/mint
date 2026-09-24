import os.path as osp
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden


@DATASET_REGISTRY.register()
class Office31Flex(DatasetBase):
    """Office-31 with flexible directory layout support for DA."""

    dataset_dir = "office31"
    domains = ["amazon", "dslr", "webcam"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS)
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS)
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS)

        super().__init__(train_x=train_x, train_u=train_u, test=test)

    def _read_data(self, input_domains):
        items = []

        for domain_idx, domain_name in enumerate(input_domains):
            domain_dir = osp.join(self.dataset_dir, domain_name)
            image_root = osp.join(domain_dir, "images")
            class_root = image_root if osp.isdir(image_root) else domain_dir

            if not osp.isdir(class_root):
                raise FileNotFoundError(
                    "Office-31 domain folder not found: "
                    f"{class_root}. Expected either '<root>/office31/{domain_name}/images/*' "
                    "or '<root>/office31/{domain_name}/*'."
                )

            class_names = listdir_nohidden(class_root)
            class_names.sort()

            for label, class_name in enumerate(class_names):
                class_dir = osp.join(class_root, class_name)
                if not osp.isdir(class_dir):
                    continue

                for image_name in listdir_nohidden(class_dir):
                    impath = osp.join(class_dir, image_name)
                    if osp.isdir(impath):
                        continue

                    items.append(
                        Datum(
                            impath=impath,
                            label=label,
                            domain=domain_idx,
                            classname=class_name,
                        )
                    )

        return items


def _normalize_office31_domain(domain_name):
    aliases = {
        "a": "amazon",
        "amazon": "amazon",
        "d": "dslr",
        "dslr": "dslr",
        "w": "webcam",
        "webcam": "webcam",
    }
    key = str(domain_name).strip().lower()
    if key not in aliases:
        raise ValueError(f"Unsupported Office-31 domain: {domain_name}")
    return aliases[key]


@DATASET_REGISTRY.register()
class Office31MTDA(DatasetBase):
    """Office-31 single-source, two-target closed-set MTDA protocol."""

    dataset_dir = "office31"
    domains = ["amazon", "dslr", "webcam"]
    domain_codes = OrderedDict(
        [("amazon", "A"), ("dslr", "D"), ("webcam", "W")]
    )

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        if not osp.isdir(self.dataset_dir):
            raise FileNotFoundError(
                f'Office31MTDA expects a dataset directory at "{self.dataset_dir}"'
            )

        source_domains = [
            _normalize_office31_domain(domain)
            for domain in cfg.DATASET.SOURCE_DOMAINS
        ]
        if len(source_domains) != 1:
            raise ValueError(
                "Office31MTDA requires exactly one source domain, "
                f"but got {source_domains}"
            )
        self.source_domain = source_domains[0]

        expected_targets = [
            domain for domain in self.domains if domain != self.source_domain
        ]
        raw_targets = list(getattr(cfg.DATASET, "TARGET_DOMAINS", []))
        if not raw_targets or raw_targets == ["auto"]:
            target_domains = expected_targets
        else:
            target_domains = [
                _normalize_office31_domain(domain) for domain in raw_targets
            ]
        if sorted(target_domains) != sorted(expected_targets):
            raise ValueError(
                "Office31MTDA expects the two remaining domains as targets. "
                f"Source={self.source_domain}, expected={expected_targets}, "
                f"got={target_domains}"
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

    def _class_root(self, domain_name):
        domain_dir = osp.join(self.dataset_dir, domain_name)
        image_root = osp.join(domain_dir, "images")
        class_root = image_root if osp.isdir(image_root) else domain_dir
        if not osp.isdir(class_root):
            raise FileNotFoundError(
                f'Missing Office-31 domain directory: "{domain_dir}"'
            )
        return class_root

    def _build_label_mapping(self, reference_domain):
        class_root = self._class_root(reference_domain)
        class_names = [
            name
            for name in listdir_nohidden(class_root)
            if osp.isdir(osp.join(class_root, name))
        ]
        class_names.sort()
        if len(class_names) != 31:
            raise ValueError(
                "Office31MTDA expects exactly 31 classes, "
                f"but found {len(class_names)} in {reference_domain}."
            )
        return OrderedDict(
            (class_name, label) for label, class_name in enumerate(class_names)
        )

    def _read_domain(self, domain_name, class_to_label):
        class_root = self._class_root(domain_name)
        class_names = [
            name
            for name in listdir_nohidden(class_root)
            if osp.isdir(osp.join(class_root, name))
        ]
        class_names.sort()
        expected_classes = list(class_to_label)
        if class_names != expected_classes:
            raise ValueError(
                "Closed-set Office-31 expects the same classes in every domain. "
                f"Domain {domain_name} does not match {self.source_domain}."
            )

        domain_index = self.domains.index(domain_name)
        items = []
        for class_name in class_names:
            class_dir = osp.join(class_root, class_name)
            for image_name in listdir_nohidden(class_dir):
                image_path = osp.join(class_dir, image_name)
                if osp.isdir(image_path):
                    continue
                items.append(
                    Datum(
                        impath=image_path,
                        label=class_to_label[class_name],
                        domain=domain_index,
                        classname=class_name.lower(),
                    )
                )
        return items
