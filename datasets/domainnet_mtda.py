import os
import os.path as osp
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


def _normalize_domain_name(domain_name):
    aliases = {
        "c": "clipart",
        "clipart": "clipart",
        "i": "infograph",
        "infograph": "infograph",
        "p": "painting",
        "painting": "painting",
        "q": "quickdraw",
        "quickdraw": "quickdraw",
        "quick_draw": "quickdraw",
        "r": "real",
        "real": "real",
        "real_world": "real",
        "s": "sketch",
        "sketch": "sketch",
    }
    key = str(domain_name).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in aliases:
        raise ValueError(f"Unsupported DomainNet domain: {domain_name}")
    return aliases[key]


@DATASET_REGISTRY.register()
class DomainNetMTDA(DatasetBase):
    """Official-list DomainNet single-source multi-target protocol."""

    dataset_dir = "DomainNet"
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    domain_codes = OrderedDict(
        (domain, code)
        for domain, code in zip(domains, ["C", "I", "P", "Q", "R", "S"])
    )

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        candidate = osp.join(root, self.dataset_dir)
        self.dataset_dir = candidate if osp.isdir(candidate) else root
        self.list_dir = osp.join(self.dataset_dir, "image_list")
        self.skip_image_file_checks = os.environ.get(
            "DOMAINNET_SKIP_IMAGE_FILE_CHECKS", ""
        ).strip().lower() in {"1", "true", "yes"}
        if not osp.isdir(self.list_dir):
            raise FileNotFoundError(
                "DomainNetMTDA expects image_list/ under either DATASET.ROOT or "
                f'DATASET.ROOT/DomainNet, but found neither below "{root}"'
            )

        source_domains = [_normalize_domain_name(d) for d in cfg.DATASET.SOURCE_DOMAINS]
        if len(source_domains) != 1:
            raise ValueError(
                "DomainNetMTDA requires exactly one source domain, "
                f"but got {source_domains}"
            )
        self.source_domain = source_domains[0]
        expected_targets = [d for d in self.domains if d != self.source_domain]
        raw_targets = list(getattr(cfg.DATASET, "TARGET_DOMAINS", []))
        target_domains = (
            expected_targets
            if not raw_targets or raw_targets == ["auto"]
            else [_normalize_domain_name(d) for d in raw_targets]
        )
        if len(target_domains) != len(set(target_domains)) or set(target_domains) != set(
            expected_targets
        ):
            raise ValueError(
                "DomainNetMTDA expects all five remaining domains exactly once. "
                f"Source={self.source_domain}, expected={expected_targets}, got={target_domains}"
            )
        self.target_domains = target_domains
        self.check_input_domains([self.source_domain], self.target_domains)

        label_to_classname = self._build_global_label_mapping()
        train_x = self._read_split(self.source_domain, "train", label_to_classname)
        train_u_by_domain = OrderedDict()
        test_by_domain = OrderedDict()
        train_u, test = [], []
        for domain in self.target_domains:
            train_items = self._read_split(domain, "train", label_to_classname)
            test_items = self._read_split(domain, "test", label_to_classname)
            train_u_by_domain[domain] = train_items
            test_by_domain[domain] = test_items
            train_u.extend(train_items)
            test.extend(test_items)

        super().__init__(train_x=train_x, train_u=train_u, test=test)
        # DatasetBase derives classnames only from train_x. The official
        # painting_train split contains no example of class 327, so that would
        # create 344 prompts while retaining raw labels up to 344. Keep the
        # canonical 345-way label space shared by every DomainNet domain.
        self._num_classes = len(label_to_classname)
        self._lab2cname = dict(label_to_classname)
        self._classnames = [
            label_to_classname[label] for label in range(self._num_classes)
        ]
        self.train_u_by_domain = train_u_by_domain
        self.test_by_domain = test_by_domain

    def _list_path(self, domain, split):
        path = osp.join(self.list_dir, f"{domain}_{split}.txt")
        if not osp.isfile(path):
            raise FileNotFoundError(f'Missing DomainNet split list: "{path}"')
        return path

    @staticmethod
    def _parse_line(line, list_path, line_number):
        try:
            relative_path, raw_label = line.rstrip().rsplit(maxsplit=1)
            return relative_path, int(raw_label)
        except ValueError as exc:
            raise ValueError(
                f"Malformed DomainNet list entry at {list_path}:{line_number}: {line!r}"
            ) from exc

    def _build_global_label_mapping(self):
        mapping = {}
        for domain in self.domains:
            for split in ("train", "test"):
                list_path = self._list_path(domain, split)
                with open(list_path, "r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        relative_path, label = self._parse_line(
                            line, list_path, line_number
                        )
                        parts = relative_path.replace("\\", "/").split("/")
                        if len(parts) < 3:
                            raise ValueError(
                                f"Cannot infer class name from {relative_path!r}"
                            )
                        classname = parts[-2].replace("_", " ").lower()
                        previous = mapping.setdefault(label, classname)
                        if previous != classname:
                            raise ValueError(
                                f"Inconsistent DomainNet label {label}: "
                                f"{previous!r} versus {classname!r}"
                            )
        expected = list(range(345))
        if sorted(mapping) != expected:
            raise ValueError(
                "DomainNet global label mapping must contain labels 0..344; "
                f"got {len(mapping)} labels"
            )
        return mapping

    def _read_split(self, domain, split, label_to_classname):
        items = []
        list_path = self._list_path(domain, split)
        domain_index = self.domains.index(domain)
        with open(list_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                relative_path, label = self._parse_line(line, list_path, line_number)
                impath = osp.join(self.dataset_dir, relative_path)
                if not self.skip_image_file_checks and not osp.isfile(impath):
                    raise FileNotFoundError(
                        f'Missing image referenced by {list_path}:{line_number}: "{impath}"'
                    )
                items.append(
                    self._make_datum(
                        impath=impath,
                        label=label,
                        domain=domain_index,
                        classname=label_to_classname[label],
                    )
                )
        return items

    def _make_datum(self, *, impath, label, domain, classname):
        if not self.skip_image_file_checks:
            return Datum(
                impath=impath,
                label=label,
                domain=domain,
                classname=classname,
            )
        datum = Datum.__new__(Datum)
        datum._impath = impath
        datum._label = label
        datum._domain = domain
        datum._classname = classname
        return datum
