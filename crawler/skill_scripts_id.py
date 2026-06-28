import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # '..', '..'代表第三级目录
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import os
import json
import logging
from pathlib import Path

import utils
from utils import get_workspace

import os
import utils
import logging
import json
from utils import get_workspace


SKILL_DATA_PATH = os.path.join(
    utils.ss_dir,
    "crawler",
    "data",
    "malicious_skills.json"
)
# 扫描SKILL.md所有捆绑的脚本.py, shell, javascript
SCRIPT_EXTENSIONS = {
    ".py",
    # ".sh", ".bash", ".fish",
    # ".js", ".mjs", ".cjs"
}

# 扫描时跳过的目录，避免噪声
SKIP_DIRS = {
    ".git", "__pycache__", ".idea", ".vscode",
    "node_modules", "dist", "build", ".next",
    ".venv", "venv", "env",
    ".pytest_cache", ".mypy_cache"
}


class Skill:
    def __init__(self, skill_id='', skill_data_path=''):
        if not skill_id:
            logging.error("Skill ID is empty.")
            raise ValueError("Skill ID is empty.")

        self.skill_id = skill_id
        self.workspace = get_workspace()
        self.repo_dir = os.path.join(self.workspace, 'repo', skill_id)

        self.skill_data_path = skill_data_path or os.path.join(
            utils.ss_dir, "crawler", "data_origin", "malicious_skills.json"
        )

        if not os.path.exists(self.skill_data_path):
            logging.error(f"Skill data file does not exist: {self.skill_data_path}")
            raise FileNotFoundError(f"Skill data file does not exist: {self.skill_data_path}")

        self.skill_data = self.get_skill_data(skill_id)
        self.repo_path = self.get_repo_path(skill_id)
        self.skill_path = self.get_skill_path(skill_id)

    def __str__(self):
        return f"Skill (ID: {self.skill_id}, RepoPath: {self.repo_path}, SkillPath: {self.skill_path})"

    def get_skill_data(self, skill_id='', skill_data_path=''):
        """
        Get skill data from skill_data_path by skill_id.
        :param skill_id: Skill ID to look for.
        :param skill_data_path: Path to the skill_data_path file.
        :return: Skill data or None if not found.
        """
        if not skill_data_path:
            skill_data_path = self.skill_data_path
        if not os.path.exists(skill_data_path):
            logging.error(f"Skill data file does not exist: {skill_data_path}")
            return None
        skill_data_path = skill_data_path
        try:
            with open(skill_data_path, 'r', encoding='utf-8') as f:
                all_skill_data = json.load(f)
            for sd in all_skill_data:
                if str(sd.get('id')) == skill_id:
                    return sd
            logging.error(f"Skill ID {skill_id} not found in {skill_data_path}.")
            return None
        except Exception as e:
            logging.error(f"Failed to read skill data ({skill_id}) from {skill_data_path}: {e}")
            return None

    @staticmethod
    def get_all_dirs_files(top_path):
        """
        Get all directories and files in the top_path.
        :param top_path: Path to the directory to scan.
        :return: Tuple of (directories list, files list).
        """
        entries = os.listdir(top_path)
        dirs = [d for d in entries if os.path.isdir(os.path.join(top_path, d))]
        files = [f for f in entries if os.path.isfile(os.path.join(top_path, f))]
        return dirs, files

    def get_repo_path(self, skill_id: str = ""):
        """
        根据 skill_id 定位已经存在的仓库目录。

        修改说明：
        - 不再尝试从 workspace/zip/<skill_id>.zip 解压。
        - 只使用 JSON 中已有 repo_path、workspace/repo/<id>、workspace/new_repo/<id>。
        - 如果以上路径都不存在，直接返回 None。
        """
        try:
            if not skill_id:
                raise ValueError("Skill ID is empty.")

            # 1) 优先使用 JSON 中已有 repo_path
            if self.skill_data and self.skill_data.get("repo_path"):
                repo_path = resolve_data_path(self.skill_data.get("repo_path"))
                if os.path.exists(repo_path):
                    return repo_path

            # 2) 再看 repo/<id>
            repo_path = os.path.join(self.workspace, "repo", skill_id)
            if os.path.exists(repo_path):
                folders, files = self.get_all_dirs_files(repo_path)
                if len(folders) == 1 and len(files) == 0:
                    repo_path = os.path.join(repo_path, folders[0])
                if os.path.exists(repo_path):
                    return repo_path

            # 3) 再看 new_repo/<id>
            repo_path = os.path.join(self.workspace, "new_repo", skill_id)
            if os.path.exists(repo_path):
                folders, files = self.get_all_dirs_files(repo_path)
                if len(folders) == 1 and len(files) == 0:
                    repo_path = os.path.join(repo_path, folders[0])
                if os.path.exists(repo_path):
                    return repo_path

            # 4) 删除 zip 解压逻辑：找不到已存在 repo 时直接返回 None
            logging.warning(
                "Repo path not found for skill ID %s. Skip extraction by design.",
                skill_id
            )
            return None

        except Exception as e:
            logging.error(f"Error getting repo path for skill ID {skill_id}: {e}")
            return None

    def get_skill_path(self, skill_id=''):
        try:
            if not skill_id:
                raise ValueError("Skill ID is empty.")

            skill_data = self.get_skill_data(skill_id)
            if not skill_data:
                raise Exception(f"Cannot get skill data for skill ID: {skill_id}")

            if skill_data.get("skill_path"):
                skill_path = os.path.normpath(skill_data.get("skill_path"))
                if os.path.exists(skill_path):
                    return skill_path

            git_source = skill_data.get('source_url', '')
            if not git_source:
                raise Exception(f"No source URL found for skill ID: {skill_id}")

            repo_path = self.get_repo_path(skill_id)
            if not repo_path:
                raise Exception(f"Cannot get repo path for skill ID: {skill_id}")

            if "/tree/" in git_source and "tree" == git_source.removeprefix("https://github.com/").split("/")[2]:
                skill_path = os.path.join(repo_path, '/'.join(git_source.split("/tree/")[1].split("/")[1:]))
            else:
                skill_path = repo_path

            skill_path = skill_path.removesuffix('/.') if skill_path.endswith('/.') else skill_path

            if not os.path.exists(skill_path):
                raise FileNotFoundError(f"Skill path does not exist: {skill_path}")

            return skill_path

        except Exception as e:
            logging.error(f"Error getting skill path for skill ID {skill_id}: {e}")
            return None



def load_skill_ids_from_json(skill_data_path: str):
    if not os.path.exists(skill_data_path):
        raise FileNotFoundError(f"Skill data file does not exist: {skill_data_path}")

    with open(skill_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids = []
    for item in data:
        skill_id = item.get("id")
        if skill_id:
            ids.append(str(skill_id))
    return ids


def is_script_file(file_path: str) -> bool:
    """
    判断文件是否可视为 script：
    1) 常见脚本后缀
    2) 无扩展名但首行带 shebang
    """
    suffix = Path(file_path).suffix.lower()
    if suffix in SCRIPT_EXTENSIONS:
        return True

    if suffix == "":
        try:
            with open(file_path, "rb") as f:
                first_line = f.readline(256)
            if first_line.startswith(b"#!"):
                return True
        except Exception:
            return False

    return False


def normalize_scan_root(skill_obj: Skill):
    """
    统一得到要扫描的根目录。
    优先级：
    1) skill.skill_path
    2) 从 repo_path 中自动回退寻找包含 SKILL.md 的目录
    3) 再不行就用 repo_path
    """
    skill_path = skill_obj.skill_path
    repo_path = skill_obj.repo_path

    # 1) 优先使用 Skill 类已经解析出的真实 skill_path
    if skill_path and os.path.exists(skill_path):
        if os.path.isdir(skill_path):
            return skill_path, "skill_path"
        if os.path.isfile(skill_path):
            return os.path.dirname(skill_path), "skill_file_parent"

    # 2) 回退：在 repo 里寻找包含 SKILL.md / skill.md 的目录
    if repo_path and os.path.exists(repo_path):
        candidates = []

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            lower_files = {f.lower() for f in files}
            if "skill.md" in lower_files:
                candidates.append(root)

        if len(candidates) == 1:
            return candidates[0], "repo_search_unique_skill_md"

        if len(candidates) > 1:
            # 尽量根据 source_url 的 /tree/ 相对路径命中
            source_url = (skill_obj.skill_data or {}).get("source_url", "")
            if "/tree/" in source_url:
                try:
                    rel_part = source_url.split("/tree/", 1)[1]
                    rel_segments = rel_part.split("/")[1:]  # 去掉 branch
                    rel_tail = os.path.join(*rel_segments) if rel_segments else ""
                    if rel_tail:
                        for c in candidates:
                            norm_c = os.path.normpath(c).lower()
                            if os.path.normpath(rel_tail).lower() in norm_c:
                                return c, "repo_search_tree_match"
                except Exception:
                    pass

            # 多个候选时，先返回第一个，同时标记来源
            return candidates[0], "repo_search_multi_skill_md_first"

        # 3) 最后兜底：整个 repo
        return repo_path, "repo_fallback"

    return None, "unresolved"


def find_script_files(scan_root: str):
    """
    递归查找脚本文件，返回相对路径列表
    """
    found = []

    for root, dirs, files in os.walk(scan_root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for filename in files:
            full_path = os.path.join(root, filename)
            if is_script_file(full_path):
                rel_path = os.path.relpath(full_path, scan_root)
                found.append(rel_path)
                # found.append(full_path)

    return sorted(set(found))


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_id_txt(path: str, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(str(item["id"]) + "\n")


def find_all_skill_roots(skill_obj: Skill):
    """
    找出同一个 repo_path 下所有包含 SKILL.md 的 skill 根目录
    返回绝对路径
    """
    repo_path = skill_obj.repo_path
    candidates = []

    # 先把 Skill 类已经解析出的主 skill_path 放进去
    if skill_obj.skill_path and os.path.exists(skill_obj.skill_path):
        if os.path.isdir(skill_obj.skill_path):
            candidates.append(os.path.abspath(os.path.normpath(skill_obj.skill_path)))
        elif os.path.isfile(skill_obj.skill_path):
            candidates.append(os.path.abspath(os.path.normpath(os.path.dirname(skill_obj.skill_path))))

    # 再扫描整个 repo，找所有包含 SKILL.md 的目录
    if repo_path and os.path.exists(repo_path):
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            lower_files = {f.lower() for f in files}
            if "skill.md" in lower_files:
                candidates.append(os.path.abspath(os.path.normpath(root)))

    candidates = sorted(set(candidates))

    if candidates:
        return candidates, "all_skill_paths"

    scan_root, path_source = normalize_scan_root(skill_obj)
    if scan_root:
        return [os.path.abspath(os.path.normpath(scan_root))], path_source

    return [], "unresolved"


def build_all_skill_path_details(skill_script_map, skill_root_map):
    """
    只输出“有脚本文件”的 skill。
    并把 skill_code_path 改成脚本文件的绝对路径。
    例如：
    [
      {
        "skill_name": "idea-refine",
        "skill_code_path": "G:\\...\\skills\\idea-refine\\scripts\\idea-refine.sh"
      }
    ]
    """
    results = []

    for field_name, rel_files in skill_script_map.items():
        if not rel_files:
            continue

        skill_root = skill_root_map.get(field_name)
        if not skill_root:
            continue

        skill_name = os.path.basename(os.path.abspath(os.path.normpath(skill_root)))

        for rel_file in rel_files:
            abs_file_path = os.path.abspath(
                os.path.normpath(os.path.join(skill_root, rel_file))
            )

            results.append({
                "skill_name": skill_name,
                "skill_code_path": abs_file_path
            })

    return results


def safe_skill_field_name(skill_name: str) -> str:
    """
    把 skill 名字转成安全的 JSON 字段名
    例如 ui-ux-pro-max -> ui_ux_pro_max
    """
    if not skill_name:
        return "unknown_skill"

    result = []
    for ch in skill_name:
        if ch.isalnum():
            result.append(ch.lower())
        else:
            result.append("_")

    name = "".join(result)
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_") or "unknown_skill"


def collect_script_files_by_skill(skill_obj: Skill):
    """
    返回：
    1) 所有 skill 根目录
    2) path_source
    3) 每个 skill 单独的 script_files 字段
    4) 有脚本的 skill_root 映射
    5) 总 script_count
    """
    skill_roots, path_source = find_all_skill_roots(skill_obj)

    if not skill_roots:
        return [], "unresolved", {}, {}, 0

    skill_script_map = {}
    skill_root_map = {}
    total_count = 0
    used_field_names = {}

    for skill_root in skill_roots:
        skill_root = os.path.abspath(os.path.normpath(skill_root))
        skill_name = os.path.basename(skill_root)
        base_field = safe_skill_field_name(skill_name)

        # 防止同名 skill 字段冲突
        used_field_names[base_field] = used_field_names.get(base_field, 0) + 1
        if used_field_names[base_field] > 1:
            field_name = f"{base_field}_{used_field_names[base_field]}_script_files"
        else:
            field_name = f"{base_field}_script_files"

        files = find_script_files(skill_root)
        skill_script_map[field_name] = files

        # 只有当前 skill 真有脚本时，才记录到 root_map
        if files:
            skill_root_map[field_name] = skill_root

        total_count += len(files)

    return skill_roots, path_source, skill_script_map, skill_root_map, total_count


def main():
    utils.setup_logging()

    output_dir = os.path.join("../crawler/data") # /root/Skill/crawler/data
    os.makedirs(output_dir, exist_ok=True)

    all_skill_ids = load_skill_ids_from_json(SKILL_DATA_PATH)

    print(f"[*] Total skills to inspect: {len(all_skill_ids)}")
    logging.info("Total skills to inspect: %d", len(all_skill_ids))

    with_scripts = []
    without_scripts = []
    errors = []

    total_skill_folder_count = 0  # 所有 Skill 文件夹数量
    total_skill_folder_with_scripts_count = 0  # 含 scripts / 脚本文件的 Skill 文件夹数量
    total_script_file_count = 0  # 所有脚本文件数量

    for idx, skill_id in enumerate(all_skill_ids, 1):
        skill_id = str(skill_id)

        try:
            skill = Skill(skill_id=skill_id, skill_data_path=SKILL_DATA_PATH)

            skill_roots, path_source, skill_script_map, skill_root_map, total_script_count = collect_script_files_by_skill(
                skill)

            if not skill_roots:
                raise FileNotFoundError(f"Cannot resolve scan root for skill {skill_id}")

            all_skill_path_details = build_all_skill_path_details(skill_script_map, skill_root_map)

            current_skill_folder_count = len(skill_roots)
            current_skill_folder_with_scripts_count = len(skill_root_map)

            # ========== 新增：累加全局统计 ==========
            total_skill_folder_count += current_skill_folder_count
            total_skill_folder_with_scripts_count += current_skill_folder_with_scripts_count
            total_script_file_count += total_script_count

            record = {
                "id": skill_id,
                "name": (skill.skill_data or {}).get("name", ""),
                "label": True, # revise
                "description": (skill.skill_data or {}).get("description", ""),
                "source_url": (skill.skill_data or {}).get("source_url", ""),
                "repo_path": os.path.abspath(os.path.normpath(skill.repo_path)) if skill.repo_path else "",
                "skill_path": os.path.abspath(os.path.normpath(skill.skill_path)) if skill.skill_path else "",
                "scan_root": os.path.abspath(os.path.normpath(skill.repo_path)) if skill.repo_path else "",
                "path_source": path_source,
                "num_skill": len(skill_roots), # 该仓库下所有 Skill 数量
                "num_skill_with_scripts": len(skill_root_map), # 该仓库下含有脚本文件的 Skill 数量
                "script_count": total_script_count,
                "all_skill_paths": all_skill_path_details
            }

            record.update(skill_script_map)

            if total_script_count > 0:
                with_scripts.append(record)
            else:
                without_scripts.append(record)

        except Exception as e:
            errors.append({
                "id": skill_id,
                "error": str(e)
            })
            logging.error("Failed to inspect skill %s: %s", skill_id, e)

        if idx % 50 == 0 or idx == len(all_skill_ids):
            print(
                f"[*] Progress {idx}/{len(all_skill_ids)} | "
                f"with_scripts={len(with_scripts)} | "
                f"without_scripts={len(without_scripts)} | "
                f"errors={len(errors)}"
            )

    save_json(os.path.join(output_dir, "malicious_skills_with_scripts.json"), with_scripts)

    logging.info("===== Done =====")
    logging.info(f"所有文件夹中的Skill数量: {total_skill_folder_count}")
    logging.info(f"含有scripts的Skill数量: {total_skill_folder_with_scripts_count}")
    logging.info(f"skills_with_scripts.json -> {os.path.join(output_dir, 'all_skills_with_scripts.json')}")

if __name__ == "__main__":
    main()
    '''
        2 
        取出抓取文件的中带有scripts的文件数据id到../data/crawler/data/all_skills_data_with_scripts
        输出所有文件夹中的Skill数量，Skill文件夹中含有scripts的Skill数量。
    '''