//! fsck — filesystem repair in the spirit of `fsck.ubifs --repair`
//! (mtd-utils `ubifs-utils/fsck.ubifs`, the `rebuild_fs` algorithm),
//! adapted to images.
//!
//! The committed index is NOT used at all: every main-area LEB is
//! scanned for valid nodes (CRC plus the semantic checks of the
//! official `parse_ino_node`/`parse_dent_node`/`parse_data_node`/
//! `parse_trun_node`), the newest sqnum copy of every inode /
//! directory entry / data block / xattr entry / truncation is kept,
//! and the file tree is reassembled from scratch — which makes the
//! rebuild work exactly when the strict open ([`crate::reader::Ubifs::new`])
//! fails on a broken index (an index branch pointing to an unmapped
//! or garbage LEB).
//!
//! There is no committed-state boundary: like the official rebuild,
//! everything physically on flash participates (`scan_nodes` filters
//! nothing by sqnum), so an uncommitted bud tail written right before
//! a power cut is recovered rather than dropped, and a readable master
//! is not needed at all — a volume with both master copies destroyed
//! rebuilds fine. The journal is not replayed through the TNC (the
//! scan absorbs the bud contents directly — the same nodes, the same
//! newest-wins rule).
//!
//! Salvage decisions mirror the official rebuild exactly:
//! - deletions are encoded on flash as an inode node with `nlink == 0`
//!   and a dent node with `inum == 0`; the newest such node wins over
//!   the live counterpart (the net effect of the official
//!   valid/deleted trees + `remove_del_nodes`);
//! - a dent is dropped when its target inode is missing
//!   (`FILE_HAS_NO_INODE`), when its type disagrees with the target
//!   (`FILE_HAS_INCONSIST_TYPE`, including a plain dent onto an xattr
//!   inode), or when it cannot be reached from `/`
//!   (`DENTRY_IS_UNREACHABLE`, with the cycle cut);
//! - a directory or xattr target keeps only its newest dent
//!   (`FILE_HAS_TOO_MANY_DENT`);
//! - a file with no surviving dent is DELETED — the official rebuild
//!   has no lost+found (that flow exists only in the non-rebuild
//!   modes, which repair a live TNC in place and do not map onto
//!   images);
//! - a regular file missing a data block keeps its size and the
//!   missing block materializes as zeros: the official
//!   `calculate_file_info` never truncates at a hole and never drops
//!   a file for a missing block (a hole is indistinguishable from a
//!   sparse hole);
//! - truncations: data blocks stale relative to a newer truncation
//!   node or a newer inode size are dropped, and the size follows
//!   `calculate_file_info` exactly (trun vs inode sqnum, then
//!   `max(ino.size, data)`);
//! - hard links stay links: the second and later dents of an inode
//!   become [`crate::tree::Kind::Hardlink`] markers plus a link group
//!   (`calc_nlink` = the dent count) — rebuild the image with
//!   [`crate::mkfs::mkfs_plan`], not plain [`crate::mkfs::mkfs`];
//! - when the root inode is missing or invalid, every file is
//!   unreachable and deleted, and a fresh empty root is created (the
//!   official `create_root`) — the rebuild never fails on a lost
//!   root; `root_created` in the report says when that happened.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::compress::decompress;
use crate::error::{Error, Result};
use crate::key::{UBIFS_DATA_KEY, UBIFS_DENT_KEY, UBIFS_INO_KEY, UBIFS_XENT_KEY};
use crate::media::*;
use crate::tree::{Inode, Kind};
use crate::walk::{LinkGroup, Plan};

/// The maximum file size addressable by the simple key format
/// (`key_max_inode_size`): 2^`UBIFS_S_KEY_BLOCK_BITS` blocks.
const MAX_INO_SZ: u64 = (1u64 << crate::key::UBIFS_S_KEY_BLOCK_BITS) * UBIFS_BLOCK_SIZE as u64;

/// What the rebuild did — logged by the caller. The counters mirror
/// the problem classes of the official `problem.c` table.
#[derive(Debug, Default, Clone)]
pub struct RebuildReport {
    /// Mapped main-area LEBs scanned.
    pub lebs_scanned: usize,
    /// Valid file nodes collected (ino/dent/xent/data/trun).
    pub nodes_scanned: usize,
    /// Nodes dropped by the semantic checks (the official `parse_*`
    /// validation: bad sizes, type inconsistencies, bad names…).
    pub invalid_nodes: usize,
    /// Failed parse attempts (dirty space resynced through).
    pub garbage_nodes: usize,
    /// Internal nodes skipped (index/log/superblock/master — not file
    /// content).
    pub internal_nodes: usize,
    /// Names removed by a newer deletion dent (inum 0).
    pub deleted_names: usize,
    /// Dents dropped: the target inode is not on flash.
    pub dropped_dangling_dents: usize,
    /// Dents dropped: the dent type disagrees with the target inode
    /// (or a plain dent points at an xattr inode).
    pub dropped_type_mismatch_dents: usize,
    /// Dents dropped by the directory/xattr single-dent rule (only the
    /// newest survives).
    pub dropped_extra_dents: usize,
    /// Dents dropped: they close a dent cycle or re-enter an already
    /// built directory.
    pub dropped_cycle_dents: usize,
    /// Files dropped: their newest inode carries `nlink == 0` (the
    /// deletion marker of the last unlink).
    pub dropped_zero_nlink_files: Vec<String>,
    /// Files deleted: no surviving dent reaches them from `/` (the
    /// official rebuild has no lost+found).
    pub dropped_unreachable_inos: usize,
    /// Xattr entries dropped (missing or non-xattr target, host gone).
    pub dropped_xattrs: usize,
    /// Data nodes of non-regular files dropped
    /// (`FILE_SHOULDNT_HAVE_DATA`).
    pub dropped_nonreg_data: usize,
    /// Data blocks dropped as stale (older than a truncation below
    /// their offset, or beyond a newer inode's size).
    pub stale_blocks_dropped: usize,
    /// Missing data blocks materialized as zeros (sparse holes and
    /// damage look the same to the official checker — informational).
    pub zero_filled_blocks: usize,
    /// No valid root inode was on flash: everything was deleted and a
    /// fresh empty root created (`create_root`).
    pub root_created: bool,
}

/// One recovered data block: the sqnum, the uncompressed size and the
/// decompressed content.
type ScannedBlock = (u64, u32, Vec<u8>);

/// Everything the scan collected, newest-sqnum-wins per identity.
#[derive(Default)]
struct Scanned {
    /// inum → the inode node.
    inos: BTreeMap<u32, InoNode>,
    /// inum → the sqnum of the kept inode node.
    ino_sqnum: BTreeMap<u32, u64>,
    /// inum → the on-flash position of the inode node (link-group
    /// ordering).
    ino_pos: BTreeMap<u32, u64>,
    /// parent inum → name → (sqnum, target inum, dent type). A target
    /// of 0 is the deletion marker for the name.
    dents: BTreeMap<u32, BTreeMap<String, (u64, u32, u8)>>,
    /// host inum → name → (sqnum, xattr inum). An inum of 0 is the
    /// deletion marker.
    xents: BTreeMap<u32, BTreeMap<String, (u64, u32)>>,
    /// inum → block → the recovered data block.
    data: BTreeMap<u32, BTreeMap<u32, ScannedBlock>>,
    /// inum → (sqnum, new size) of the newest truncation node.
    truns: BTreeMap<u32, (u64, u64)>,
}

/// The on-flash type of a mode (`ubifs_get_dent_type`).
fn itype_of(mode: u32) -> u8 {
    match mode & 0o170000 {
        0o100000 => UBIFS_ITYPE_REG,
        0o040000 => UBIFS_ITYPE_DIR,
        0o120000 => UBIFS_ITYPE_LNK,
        0o060000 => UBIFS_ITYPE_BLK,
        0o020000 => UBIFS_ITYPE_CHR,
        0o010000 => UBIFS_ITYPE_FIFO,
        _ => UBIFS_ITYPE_SOCK,
    }
}

/// The semantic checks behind the official `parse_ino_node` (the CRC
/// and the length bounds are already verified by `parse_node`).
fn ino_valid(node: &[u8], ino: &InoNode) -> bool {
    if ino.key.inum == 0 || ino.key.ty() != UBIFS_INO_KEY {
        return false;
    }
    // The root must be a directory.
    if ino.key.inum == UBIFS_ROOT_INO && ino.mode & 0o170000 != 0o040000 {
        return false;
    }
    if ino.size > MAX_INO_SZ || ino.compr_type > UBIFS_COMPR_ZSTD {
        return false;
    }
    let data_len = ino.data.len();
    if data_len > UBIFS_MAX_INO_DATA || node.len() != UBIFS_INO_NODE_SZ + data_len {
        return false;
    }
    // Encrypted inodes only exist on format-5 volumes, refused at the
    // superblock below (`inode_can_be_encrypted`).
    if ino.flags & UBIFS_CRYPT_FL != 0 {
        return false;
    }
    let is_xattr = ino.flags & UBIFS_XATTR_FL != 0;
    if is_xattr {
        // An xattr inode is a regular file whose value is its inline
        // data, with zero xattr accounting of its own.
        return ino.mode & 0o170000 == 0o100000
            && data_len == ino.size as usize
            && ino.xattr_cnt == 0
            && ino.xattr_size == 0
            && ino.xattr_names == 0;
    }
    match ino.mode & 0o170000 {
        // Regular files carry no inline data (data nodes only),
        // directories nothing at all.
        0o100000 | 0o040000 => data_len == 0,
        // A symlink target is always inline.
        0o120000 => data_len > 0,
        // A device number is the glibc makedev encoding (4 or 8 bytes).
        0o020000 | 0o060000 => data_len == 4 || data_len == 8,
        0o010000 | 0o140000 => data_len == 0,
        _ => false,
    }
}

/// The semantic checks behind the official `parse_dent_node` for both
/// dents and xattr entries (`want_ty` is the key type the node must
/// carry). `node` is the whole node; the name must end with a NUL that
/// is part of the node length.
fn dent_valid(node: &[u8], dent: &DentNode, want_ty: u32) -> bool {
    let nlen = dent.name.len();
    if dent.key.ty() != want_ty || nlen > UBIFS_MAX_NLEN || dent.itype > UBIFS_ITYPE_SOCK {
        return false;
    }
    if dent.inum > u32::MAX as u64 {
        return false; // INUM_WATERMARK
    }
    if node.len() != UBIFS_DENT_NODE_SZ + nlen + 1 || node[UBIFS_DENT_NODE_SZ + nlen] != 0 {
        return false;
    }
    // Xattr names cannot contain interior NULs.
    if want_ty == UBIFS_XENT_KEY && dent.name.contains(&0) {
        return false;
    }
    true
}

/// The semantic checks behind the official `parse_trun_node`
/// (`struct ubifs_trun_node`: ch | inum | pad | old_size | new_size).
fn trun_valid(node: &[u8]) -> Option<(u32, u64)> {
    if node.len() < UBIFS_TRUN_NODE_SZ {
        return None;
    }
    let inum = le32(node, 24);
    let old_size = le64(node, 32);
    let new_size = le64(node, 40);
    if inum == 0 || old_size > MAX_INO_SZ || new_size > MAX_INO_SZ || old_size <= new_size {
        return None;
    }
    Some((inum, new_size))
}

/// Rebuild the filesystem from the scanned LEBs. `lebs` are the volume
/// LEBs (UBI headers stripped — exactly [`crate::reader::Ubifs::new`]'s
/// input). Only the superblock must be readable (it locates the
/// areas — the same fatal contract as the official fsck); a damaged
/// master, log or index is irrelevant to the rebuild. The result is a
/// [`Plan`] so that hard links survive — feed it to
/// [`crate::mkfs::mkfs_plan`].
pub fn rebuild(lebs: &[Option<Vec<u8>>]) -> Result<(Plan, RebuildReport)> {
    let mut rep = RebuildReport::default();

    // ---- Step 0: the superblock locates the areas (fatal when
    // unreadable, exactly like the official fsck). The master is NOT
    // needed — the rebuild never consults the committed state.
    let Some(Some(sb_leb)) = lebs.first() else {
        return Err(Error::MissingData("leb 0 (superblock) is unmapped".into()));
    };
    let (node, _) = parse_node(sb_leb, sb_leb.len(), "superblock")?;
    if node[20] != UBIFS_SB_NODE {
        return Err(Error::BadNode("leb 0 is not a superblock".into()));
    }
    let sb = Superblock::parse(node)?;
    // The same refusals as the reader (the official fsck refuses
    // authenticated volumes too).
    if sb.flags & UBIFS_FLG_AUTHENTICATION != 0 {
        return Err(Error::Unsupported(
            "authenticated volumes are not supported".into(),
        ));
    }
    if sb.flags & (UBIFS_FLG_ENCRYPTION | UBIFS_FLG_DOUBLE_HASH) != 0 {
        return Err(Error::Unsupported(
            "encrypted (format 5) volumes are not supported".into(),
        ));
    }
    let main_first = UBIFS_LOG_LNUM + sb.log_lebs + sb.lpt_lebs + sb.orph_lebs;

    // ---- Step 1: scan every main-area LEB (valid/deleted split and
    // the sqnum boundary of the official `scan_nodes` are implicit
    // here: the deletion markers simply participate in the
    // newest-wins dedup, and nothing is filtered by sqnum).
    let mut sc = Scanned::default();
    for (lnum, leb) in lebs.iter().enumerate() {
        if (lnum as u32) < main_first {
            continue; // the log/LPT/orphan areas hold no file nodes
        }
        let Some(leb) = leb else { continue };
        rep.lebs_scanned += 1;
        let mut pos = 0usize;
        loop {
            if pos + 20 > leb.len() {
                break;
            }
            // erased space: skip 8 bytes at a time (a node never starts
            // with 0xFF, and the alignment fillers are 0xFF too)
            if leb[pos..pos + 4] == [0xFF; 4] {
                pos += 8;
                continue;
            }
            match parse_node(&leb[pos..], leb.len() - pos, "fsck scan") {
                Ok((node, ch)) => {
                    if ch.len == 0 {
                        pos += 8;
                        continue;
                    }
                    let sqnum = ch.sqnum;
                    let file_node = matches!(
                        ch.node_type,
                        UBIFS_INO_NODE
                            | UBIFS_DENT_NODE
                            | UBIFS_XENT_NODE
                            | UBIFS_DATA_NODE
                            | UBIFS_TRUN_NODE
                    );
                    let mut ok = false;
                    match ch.node_type {
                        UBIFS_INO_NODE => {
                            if let Ok(ino) = InoNode::parse(node) {
                                ok = ino_valid(node, &ino);
                                if ok {
                                    let inum = ino.key.inum;
                                    if sc.ino_sqnum.get(&inum).is_none_or(|s| sqnum > *s) {
                                        sc.inos.insert(inum, ino);
                                        sc.ino_sqnum.insert(inum, sqnum);
                                        sc.ino_pos.insert(inum, (lnum as u64) << 32 | pos as u64);
                                    }
                                }
                            }
                        }
                        UBIFS_DENT_NODE | UBIFS_XENT_NODE => {
                            let want = if ch.node_type == UBIFS_DENT_NODE {
                                UBIFS_DENT_KEY
                            } else {
                                UBIFS_XENT_KEY
                            };
                            if let Ok(dent) = DentNode::parse(node) {
                                ok = dent_valid(node, &dent, want);
                                if ok {
                                    let name = String::from_utf8_lossy(&dent.name).into_owned();
                                    let parent = dent.key.inum;
                                    let entry = (sqnum, dent.inum as u32, dent.itype);
                                    if want == UBIFS_DENT_KEY {
                                        let fresh = sc
                                            .dents
                                            .get(&parent)
                                            .and_then(|m| m.get(&name))
                                            .is_none_or(|(s, _, _)| sqnum > *s);
                                        if fresh {
                                            sc.dents.entry(parent).or_default().insert(name, entry);
                                        }
                                    } else {
                                        let fresh = sc
                                            .xents
                                            .get(&parent)
                                            .and_then(|m| m.get(&name))
                                            .is_none_or(|(s, _)| sqnum > *s);
                                        if fresh {
                                            sc.xents
                                                .entry(parent)
                                                .or_default()
                                                .insert(name, (entry.0, entry.1));
                                        }
                                    }
                                }
                            }
                        }
                        UBIFS_DATA_NODE => {
                            if let Ok(dn) = DataNode::parse(node) {
                                ok = dn.key.ty() == UBIFS_DATA_KEY
                                    && dn.key.inum != 0
                                    && dn.size >= 1
                                    && dn.size as usize <= UBIFS_BLOCK_SIZE
                                    && dn.compr_type <= UBIFS_COMPR_ZSTD;
                                if ok {
                                    if let Ok(content) =
                                        decompress(dn.compr_type, &dn.data, dn.size as usize)
                                    {
                                        let (inum, block) = (dn.key.inum, dn.key.block_or_hash);
                                        let fresh = sc
                                            .data
                                            .get(&inum)
                                            .and_then(|m| m.get(&block))
                                            .is_none_or(|(s, _, _)| sqnum > *s);
                                        if fresh {
                                            sc.data
                                                .entry(inum)
                                                .or_default()
                                                .insert(block, (sqnum, dn.size, content));
                                        }
                                    } else {
                                        ok = false;
                                    }
                                }
                            }
                        }
                        UBIFS_TRUN_NODE => {
                            if let Some((inum, new_size)) = trun_valid(node) {
                                ok = true;
                                if sc.truns.get(&inum).is_none_or(|(s, _)| sqnum > *s) {
                                    sc.truns.insert(inum, (sqnum, new_size));
                                }
                            }
                        }
                        UBIFS_PAD_NODE => {
                            // a pad node consumes its declared span (the
                            // filler inside is part of the pad) — the
                            // same rule the kernel scanner uses
                            let pad_len =
                                u32::from_le_bytes(node[24..28].try_into().unwrap()) as usize;
                            pos += UBIFS_PAD_NODE_SZ + pad_len;
                            continue;
                        }
                        _ => rep.internal_nodes += 1,
                    }
                    if ok {
                        rep.nodes_scanned += 1;
                    } else if file_node {
                        rep.invalid_nodes += 1;
                    }
                    pos = (pos + ch.len as usize).div_ceil(8) * 8;
                }
                Err(_) => {
                    // dirty space: the committed index references nodes
                    // at exact offsets and can point past a dirty
                    // region, so the scan resyncs at the next 8-byte
                    // boundary instead of giving up on the LEB
                    rep.garbage_nodes += 1;
                    pos += 8;
                }
            }
        }
    }

    // ---- Step 2 (remove_del_nodes): implicit — the deletion markers
    // (nlink-0 inodes, inum-0 dents) already won the newest-wins dedup
    // above, which is exactly the official valid−deleted subtraction.
    for entries in sc.dents.values() {
        rep.deleted_names += entries.values().filter(|(_, t, _)| *t == 0).count();
    }

    // ---- Step 4 (filter_invalid_files), the dent side: count the
    // incoming dents per target, then keep only the newest dent of
    // every directory/xattr target (FILE_HAS_TOO_MANY_DENT).
    let mut incoming: BTreeMap<u32, Vec<(u32, String, u64)>> = BTreeMap::new();
    for (parent, entries) in &sc.dents {
        for (name, (sqnum, target, _)) in entries {
            if *target != 0 {
                incoming
                    .entry(*target)
                    .or_default()
                    .push((*parent, name.clone(), *sqnum));
            }
        }
    }
    for (target, list) in &incoming {
        let dir_or_xattr = sc
            .inos
            .get(target)
            .is_some_and(|i| i.flags & UBIFS_XATTR_FL != 0 || i.mode & 0o170000 == 0o040000);
        if dir_or_xattr && list.len() > 1 {
            let keep = list.iter().max_by_key(|(_, _, sq)| *sq).unwrap();
            for (parent, name, _) in list {
                if (parent, name) != (&keep.0, &keep.1) {
                    sc.dents.get_mut(parent).unwrap().remove(name);
                    rep.dropped_extra_dents += 1;
                }
            }
        }
    }

    // ---- The root: a valid directory with a live nlink that is
    // nobody's child (FILE_ROOT_HAS_DENT invalidates it, exactly like
    // the official file_is_valid).
    let root_ok = sc
        .inos
        .get(&UBIFS_ROOT_INO)
        .is_some_and(|ino| ino.mode & 0o170000 == 0o040000 && ino.nlink != 0)
        && !incoming.contains_key(&UBIFS_ROOT_INO);

    // ---- Steps 3-5 (assemble the reachable tree; unreachable files
    // are deleted — the official rebuild has no lost+found).
    /// The mutable state of the tree assembly.
    struct Assemble<'a> {
        sc: &'a Scanned,
        rep: &'a mut RebuildReport,
        built_dirs: HashSet<u32>,
        /// First-occurrence path of every built non-directory.
        first_path: HashMap<u32, Vec<String>>,
        /// Surviving dents per non-directory inode (`calc_nlink`).
        dent_count: HashMap<u32, u32>,
        attached_xattrs: HashSet<u32>,
        /// The directories on the current DFS path — the dent-cycle cut.
        on_stack: HashSet<u32>,
        path: Vec<String>,
    }

    /// The shared inode fields (everything but the kind and xattrs).
    fn inode_from(ino: &InoNode, kind: Kind, xattrs: Vec<(String, Vec<u8>)>) -> Inode {
        Inode {
            kind,
            xattrs,
            nlink: 1,
            mode: ino.mode,
            uid: ino.uid,
            gid: ino.gid,
            atime_sec: ino.atime_sec,
            ctime_sec: ino.ctime_sec,
            mtime_sec: ino.mtime_sec,
            atime_nsec: ino.atime_nsec,
            ctime_nsec: ino.ctime_nsec,
            mtime_nsec: ino.mtime_nsec,
            flags: ino.flags,
        }
    }

    impl Assemble<'_> {
        /// Build the subtree of `inum` (a directory) — the DFS over the
        /// recovered dents.
        fn build_dir(&mut self, inum: u32) -> Result<Inode> {
            let sc = self.sc;
            let ino = sc
                .inos
                .get(&inum)
                .ok_or_else(|| Error::NotFound(format!("inode {inum} missing")))?;
            let entries: Vec<(String, u32, u8)> = sc
                .dents
                .get(&inum)
                .map(|m| m.iter().map(|(n, (_, t, i))| (n.clone(), *t, *i)).collect())
                .unwrap_or_default();
            let mut children: Vec<(String, Inode)> = Vec::new();
            for (name, target, itype) in entries {
                if target == 0 {
                    continue; // the deletion marker — counted at scan time
                }
                let Some(tino) = sc.inos.get(&target) else {
                    // FILE_HAS_NO_INODE: the dent points at nothing.
                    self.rep.dropped_dangling_dents += 1;
                    continue;
                };
                // FILE_HAS_INCONSIST_TYPE: the dent type must match the
                // target, and a plain dent cannot name an xattr inode.
                if itype_of(tino.mode) != itype || tino.flags & UBIFS_XATTR_FL != 0 {
                    self.rep.dropped_type_mismatch_dents += 1;
                    continue;
                }
                // FILE_HAS_0_NLINK_INODE: the newest inode says the file
                // was deleted.
                if tino.nlink == 0 {
                    self.path.push(name.clone());
                    self.rep
                        .dropped_zero_nlink_files
                        .push(format!("/{}", self.path.join("/")));
                    self.path.pop();
                    continue;
                }
                if tino.mode & 0o170000 == 0o040000 {
                    // DENTRY_IS_UNREACHABLE: a dent cycle, or a second
                    // dent onto an already built directory.
                    if self.on_stack.contains(&target) || self.built_dirs.contains(&target) {
                        self.rep.dropped_cycle_dents += 1;
                        continue;
                    }
                    self.on_stack.insert(target);
                    self.path.push(name.clone());
                    let child = self.build_dir(target);
                    self.path.pop();
                    self.on_stack.remove(&target);
                    self.built_dirs.insert(target);
                    children.push((name.clone(), child?));
                } else {
                    // Hard links: the second and later dents become
                    // markers; the link group counts the dents.
                    if let Some(first) = self.first_path.get(&target).cloned() {
                        *self.dent_count.entry(target).or_insert(0) += 1;
                        children.push((
                            name.clone(),
                            Inode::with_kind(tino.mode, Kind::Hardlink { target: first }),
                        ));
                    } else {
                        let mut p = self.path.clone();
                        p.push(name.clone());
                        self.first_path.insert(target, p);
                        *self.dent_count.entry(target).or_insert(0) += 1;
                        children.push((name.clone(), self.build_non_dir(target)?));
                    }
                }
            }
            children.sort_by(|a, b| a.0.cmp(&b.0));
            let xattrs = self.host_xattrs(inum);
            Ok(inode_from(ino, Kind::Dir(children), xattrs))
        }

        /// Build a non-directory file of `inum` (its first dent already
        /// counted).
        fn build_non_dir(&mut self, inum: u32) -> Result<Inode> {
            let sc = self.sc;
            let ino = sc
                .inos
                .get(&inum)
                .ok_or_else(|| Error::NotFound(format!("inode {inum} missing")))?;
            let mode_t = ino.mode & 0o170000;
            // FILE_SHOULDNT_HAVE_DATA: only regular files keep their
            // data nodes (xattr values are inline in the inode).
            if mode_t != 0o100000 || ino.flags & UBIFS_XATTR_FL != 0 {
                if let Some(blocks) = sc.data.get(&inum) {
                    self.rep.dropped_nonreg_data += blocks.len();
                }
            }
            let kind = match mode_t {
                0o100000 => Kind::File(self.file_content(inum, ino)),
                0o120000 => Kind::Symlink(ino.data.clone()),
                0o020000 | 0o060000 => {
                    let mut rdev = 0u64;
                    if ino.data.len() >= 4 {
                        rdev = u32::from_le_bytes(ino.data[..4].try_into().unwrap()) as u64;
                    }
                    if ino.data.len() >= 8 {
                        rdev = u64::from_le_bytes(ino.data[..8].try_into().unwrap());
                    }
                    Kind::Dev { rdev }
                }
                0o010000 => Kind::Fifo,
                _ => Kind::Socket,
            };
            let xattrs = self.host_xattrs(inum);
            Ok(inode_from(ino, kind, xattrs))
        }

        /// The content and the final size of a regular file — a
        /// faithful port of `calculate_file_info`'s data/truncation
        /// handling. Holes become zeros; nothing is truncated at a
        /// hole and no file is dropped for one.
        fn file_content(&mut self, inum: u32, ino: &InoNode) -> Vec<u8> {
            let empty = BTreeMap::new();
            let blocks = self.sc.data.get(&inum).unwrap_or(&empty);
            let (trun_sqnum, trun_size) = self
                .sc
                .truns
                .get(&inum)
                .map(|(s, n)| (*s, *n))
                .unwrap_or((0, 0));
            let ino_sqnum = self.sc.ino_sqnum.get(&inum).copied().unwrap_or(0);
            let mut new_size = 0u64;
            let mut kept: Vec<(u32, &Vec<u8>)> = Vec::new();
            for (block, (d_sqnum, dn_size, content)) in blocks {
                let d_sz = *block as u64 * UBIFS_BLOCK_SIZE as u64 + *dn_size as u64;
                // A block older than a truncation below its offset, or
                // beyond the size of a newer inode, is stale truncation
                // leftovers — dropped.
                let stale = (trun_sqnum > *d_sqnum && trun_size < d_sz)
                    || (ino_sqnum > *d_sqnum && ino.size < d_sz);
                if stale {
                    self.rep.stale_blocks_dropped += 1;
                } else {
                    new_size = new_size.max(d_sz);
                    kept.push((*block, content));
                }
            }
            let size = if trun_sqnum > ino_sqnum && trun_size < ino.size {
                if trun_size < new_size {
                    // Appends after the truncation survived on their own.
                    new_size
                } else {
                    // Only overwrites past the truncation point survived.
                    trun_size
                }
            } else {
                // The calculated size may only extend, never shrink (a
                // set i_size larger than the data is legitimate).
                ino.size.max(new_size)
            };
            let mut out = vec![0u8; size as usize];
            self.rep.zero_filled_blocks +=
                (size.div_ceil(UBIFS_BLOCK_SIZE as u64) as usize).saturating_sub(kept.len());
            for (block, content) in kept {
                let off = block as usize * UBIFS_BLOCK_SIZE;
                let end = (off + content.len()).min(out.len());
                out[off..end].copy_from_slice(&content[..end - off]);
            }
            out
        }

        /// The xattrs of `inum` from the recovered xent entries.
        fn host_xattrs(&mut self, inum: u32) -> Vec<(String, Vec<u8>)> {
            let entries: Vec<(String, u32)> = self
                .sc
                .xents
                .get(&inum)
                .map(|m| m.iter().map(|(n, (_, a))| (n.clone(), *a)).collect())
                .unwrap_or_default();
            let mut out = Vec::new();
            for (name, attr) in entries {
                // XATTR_HAS_NO_HOST / a dangling or non-xattr target.
                let Some(aino) = self.sc.inos.get(&attr) else {
                    self.rep.dropped_xattrs += 1;
                    continue;
                };
                if aino.flags & UBIFS_XATTR_FL == 0 {
                    self.rep.dropped_xattrs += 1;
                    continue;
                }
                self.attached_xattrs.insert(attr);
                // The value is the inline data of the xattr inode
                // (validated to equal its size at scan time).
                out.push((name, aino.data.clone()));
            }
            out
        }
    }

    let mut asm = Assemble {
        sc: &sc,
        rep: &mut rep,
        built_dirs: HashSet::new(),
        first_path: HashMap::new(),
        dent_count: HashMap::new(),
        attached_xattrs: HashSet::new(),
        on_stack: HashSet::new(),
        path: Vec::new(),
    };
    let root = if root_ok {
        asm.on_stack.insert(UBIFS_ROOT_INO);
        let root = asm.build_dir(UBIFS_ROOT_INO)?;
        asm.built_dirs.insert(UBIFS_ROOT_INO);
        root
    } else {
        // Every file is unreachable and deleted; a fresh empty root
        // takes their place (the official create_root).
        asm.rep.root_created = true;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        Inode::dir(0o755, Vec::new()).with_times(now)
    };
    let Assemble {
        rep,
        built_dirs,
        first_path,
        dent_count,
        attached_xattrs,
        ..
    } = asm;

    // ---- The leftover accounting: files nothing reached, and xattr
    // inodes whose host is gone (XATTR_HAS_NO_HOST).
    for (inum, ino) in &sc.inos {
        if *inum == UBIFS_ROOT_INO || ino.nlink == 0 {
            continue; // the root is built; nlink-0 is the deletion marker
        }
        if ino.flags & UBIFS_XATTR_FL != 0 {
            if !attached_xattrs.contains(inum) {
                rep.dropped_xattrs += 1;
            }
            continue;
        }
        if !built_dirs.contains(inum) && !first_path.contains_key(inum) {
            rep.dropped_unreachable_inos += 1;
        }
    }

    // ---- Step 6 (check_and_correct_files): link groups for the
    // hard-linked files, nlink recomputed from the dents.
    let mut links: HashMap<Vec<String>, u32> = HashMap::new();
    let mut link_groups = Vec::new();
    for (inum, count) in &dent_count {
        if *count > 1 {
            let tree_path = first_path[inum].clone();
            links.insert(tree_path.clone(), *count);
            link_groups.push(LinkGroup {
                tree_path,
                nlink: *count,
                order_key: sc.ino_pos.get(inum).copied().unwrap_or(0),
            });
        }
    }
    link_groups.sort_by_key(|g| g.order_key);

    /// Dirs: 2 + subdirectories; files: their dent count (1 unless
    /// hard-linked) — `calc_nlink` exactly.
    fn fix_nlink(node: &mut Inode, path: &mut Vec<String>, links: &HashMap<Vec<String>, u32>) {
        if let Kind::Dir(children) = &mut node.kind {
            let mut dirs = 0u32;
            for (name, child) in children.iter_mut() {
                path.push(name.clone());
                fix_nlink(child, path, links);
                path.pop();
                if matches!(child.kind, Kind::Dir(_)) {
                    dirs += 1;
                }
            }
            node.nlink = 2 + dirs;
        } else if let Some(n) = links.get(path) {
            node.nlink = *n;
        } else if !matches!(node.kind, Kind::Hardlink { .. }) {
            node.nlink = 1;
        }
    }
    let mut root = root;
    let mut fix_path = Vec::new();
    fix_nlink(&mut root, &mut fix_path, &links);

    Ok((Plan { root, link_groups }, std::mem::take(rep)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key::Key;
    use crate::mkfs;
    use crate::tree::Kind;
    use crate::walk::LinkGroup;

    fn geom_options() -> mkfs::Options {
        let mut o = mkfs::Options::new(2048, 126976, 2212);
        o.log_lebs = Some(5);
        o.max_bud_bytes = Some(8 << 20);
        o
    }

    fn sample_tree() -> Inode {
        Inode::dir(
            0o755,
            vec![
                (
                    "etc".into(),
                    Inode::dir(
                        0o755,
                        vec![
                            ("marker".into(), Inode::file(0o644, b"marker\n".to_vec())),
                            (
                                "big".into(),
                                Inode::file(
                                    0o600,
                                    (0..400_000u32).map(|i| (i % 251) as u8).collect(),
                                ),
                            ),
                        ],
                    ),
                ),
                (
                    "home".into(),
                    Inode::dir(
                        0o700,
                        vec![
                            (
                                "user".into(),
                                Inode::dir(
                                    0o755,
                                    vec![("link".into(), Inode::symlink("/etc/marker"))],
                                ),
                            ),
                            (
                                "big2".into(),
                                Inode::file(
                                    0o600,
                                    (0..400_000u32).map(|i| (i % 249) as u8).collect(),
                                ),
                            ),
                        ],
                    ),
                ),
                ("small.txt".into(), Inode::file(0o644, b"hello".to_vec())),
            ],
        )
    }

    /// Sort dir children recursively so a recovered tree compares equal
    /// to the reader's view (rebuild sorts by name, mkfs wrote insertion
    /// order).
    fn canonical(node: &mut Inode) {
        if let Kind::Dir(children) = &mut node.kind {
            children.sort_by(|a, b| a.0.cmp(&b.0));
            for (_, child) in children.iter_mut() {
                canonical(child);
            }
        }
    }

    /// A compact recursive fingerprint (kind, mode, size/content
    /// digest) — enough to compare trees without printing megabytes.
    fn fingerprint(node: &Inode) -> String {
        let body = match &node.kind {
            Kind::Dir(children) => {
                let inner: Vec<String> = children
                    .iter()
                    .map(|(n, c)| format!("{n}={}", fingerprint(c)))
                    .collect();
                format!("dir[{}]", inner.join(","))
            }
            Kind::File(data) => {
                let mut h: u64 = 1469598103934665603;
                for &b in data {
                    h = (h ^ b as u64).wrapping_mul(1099511628211);
                }
                format!("file({}, {h:#x})", data.len())
            }
            Kind::Symlink(t) => format!("link({})", String::from_utf8_lossy(t)),
            Kind::Dev { rdev } => format!("dev({rdev})"),
            Kind::Fifo => "fifo".into(),
            Kind::Socket => "sock".into(),
            Kind::Hardlink { target } => format!("hard({})", target.join("/")),
        };
        format!("{:o}/{}/{}", node.mode, node.nlink, body)
    }

    /// Look a node up by predicate, brute-force over the committed
    /// prefixes: returns (leb index, offset, length).
    fn node_span(
        image: &mkfs::Image,
        mut pred: impl FnMut(&[u8], &CommonHeader) -> bool,
    ) -> Option<(usize, usize, usize)> {
        for (idx, leb) in image.lebs.iter().enumerate() {
            let Some(leb) = leb else { continue };
            let mut pos = 0usize;
            while pos + 20 <= leb.len() && !leb[pos..pos + 20].iter().all(|&b| b == 0xFF) {
                let Ok((node, ch)) = parse_node(&leb[pos..], leb.len() - pos, "t") else {
                    break;
                };
                if pred(node, &ch) {
                    return Some((idx, pos, ch.len as usize));
                }
                if ch.node_type == UBIFS_PAD_NODE {
                    let pad_len = u32::from_le_bytes(node[24..28].try_into().unwrap()) as usize;
                    pos += UBIFS_PAD_NODE_SZ + pad_len;
                    continue;
                }
                pos = (pos + ch.len as usize).div_ceil(8) * 8;
            }
        }
        None
    }

    fn dent_span(image: &mkfs::Image, name: &str) -> Option<(usize, usize, usize)> {
        node_span(image, |node, ch| {
            ch.node_type == UBIFS_DENT_NODE
                && DentNode::parse(node).map(|d| d.name == name.as_bytes()) == Ok(true)
        })
    }

    /// The inum a dent with this name points at.
    fn dent_target(image: &mkfs::Image, name: &str) -> u32 {
        let (idx, pos, len) = dent_span(image, name).expect("dent");
        let leb = image.lebs[idx].as_ref().unwrap();
        let (node, _) = parse_node(&leb[pos..pos + len], len, "t").unwrap();
        DentNode::parse(node).unwrap().inum as u32
    }

    /// The (leb, offset, length) of the DATA node of `inum` block `k`.
    fn data_span(image: &mkfs::Image, inum: u32, block: u32) -> Option<(usize, usize, usize)> {
        node_span(image, |node, ch| {
            ch.node_type == UBIFS_DATA_NODE
                && DataNode::parse(node).map(|d| d.key.inum == inum && d.key.block_or_hash == block)
                    == Ok(true)
        })
    }

    /// Remove one node from its LEB together with its alignment filler
    /// (the next node must stay 8-aligned — an unaligned node is an
    /// invalid UBIFS layout), shift the tail left and FF-fill the end:
    /// the index still points at the old offset, so the strict open
    /// refuses, while the rebuild loses exactly that node.
    fn excise(image: &mkfs::Image, leb_idx: usize, pos: usize, len: usize) -> Vec<Option<Vec<u8>>> {
        // Remove the node together with its alignment filler (the next
        // node must stay 8-aligned). `drain` shortens the LEB — the
        // vacated tail is gone with it; an extra FF-fill here would
        // corrupt the payload of the node that now ends at the tail.
        let end = (pos + len).div_ceil(8) * 8;
        let mut lebs = image.lebs.clone();
        let leb = lebs[leb_idx].as_mut().unwrap();
        leb.drain(pos..end);
        lebs
    }

    /// The master node's sqnum — the committed-state boundary the
    /// strict open uses (the rebuild deliberately does not).
    fn max_sqnum(image: &mkfs::Image) -> u64 {
        let l = image.lebs[UBIFS_MST_LNUM as usize].as_ref().unwrap();
        let (node, _) = parse_node(l, l.len(), "m").unwrap();
        assert_eq!(node[20], UBIFS_MST_NODE);
        u64::from_le_bytes(node[8..16].try_into().unwrap())
    }

    /// Append a crafted node right after the committed prefix of the
    /// first content LEB (like a bud tail would be).
    fn append_node(all: &[Option<Vec<u8>>], node: &[u8]) -> Vec<Option<Vec<u8>>> {
        let idx = (2..all.len())
            .find(|&i| {
                all[i].as_ref().is_some_and(|leb| {
                    matches!(
                        parse_node(leb, leb.len(), "t").map(|(_, ch)| ch.node_type),
                        Ok(UBIFS_INO_NODE | UBIFS_DENT_NODE | UBIFS_DATA_NODE)
                    )
                })
            })
            .expect("a content LEB");
        let mut lebs = all.to_vec();
        let leb = lebs[idx].as_mut().unwrap();
        let mut pos = 0usize;
        while pos + 20 <= leb.len() && !leb[pos..pos + 20].iter().all(|&b| b == 0xFF) {
            let Ok((_, ch)) = parse_node(&leb[pos..], leb.len() - pos, "t") else {
                break;
            };
            if ch.node_type == UBIFS_PAD_NODE {
                let pad_len =
                    u32::from_le_bytes(leb[pos + 24..pos + 28].try_into().unwrap()) as usize;
                pos += UBIFS_PAD_NODE_SZ + pad_len;
                continue;
            }
            pos = (pos + ch.len as usize).div_ceil(8) * 8;
        }
        assert!(pos + node.len() <= leb.len());
        leb[pos..pos + node.len()].copy_from_slice(node);
        lebs
    }

    /// A crafted truncation node (`struct ubifs_trun_node`).
    fn trun_node(inum: u32, old_size: u64, new_size: u64, sqnum: u64) -> Vec<u8> {
        let mut n = vec![0u8; UBIFS_TRUN_NODE_SZ];
        n[16..20].copy_from_slice(&(UBIFS_TRUN_NODE_SZ as u32).to_le_bytes());
        n[24..28].copy_from_slice(&inum.to_le_bytes());
        n[32..40].copy_from_slice(&old_size.to_le_bytes());
        n[40..48].copy_from_slice(&new_size.to_le_bytes());
        finish_node(&mut n, UBIFS_TRUN_NODE, sqnum);
        n
    }

    fn crafted_ino(inum: u32, mode: u32, nlink: u32, sqnum: u64) -> Vec<u8> {
        InoNode {
            key: Key::ino(inum),
            creat_sqnum: 1,
            size: 0,
            atime_sec: 0,
            ctime_sec: 0,
            mtime_sec: 0,
            atime_nsec: 0,
            ctime_nsec: 0,
            mtime_nsec: 0,
            nlink,
            uid: 0,
            gid: 0,
            mode,
            flags: 0,
            xattr_cnt: 0,
            xattr_size: 0,
            xattr_names: 0,
            compr_type: UBIFS_COMPR_NONE,
            data: Vec::new(),
        }
        .to_node(sqnum)
    }

    fn crafted_dent(parent: u32, name: &str, inum: u32, itype: u8, sqnum: u64) -> Vec<u8> {
        DentNode {
            key: Key::dent(parent, name.as_bytes()).unwrap(),
            inum: inum as u64,
            itype,
            cookie: 0,
            name: name.as_bytes().to_vec(),
        }
        .to_node(sqnum)
    }

    fn child_mut<'a>(node: &'a mut Inode, name: &str) -> &'a mut Inode {
        let Kind::Dir(children) = &mut node.kind else {
            panic!("not a dir")
        };
        &mut children.iter_mut().find(|(n, _)| n == name).unwrap().1
    }

    #[test]
    fn broken_index_leb_is_recovered_without_it() {
        let image = mkfs::mkfs(&sample_tree(), &geom_options()).unwrap();
        // strict open must succeed on the intact image
        assert!(crate::reader::Ubifs::new(image.lebs.clone()).is_ok());

        // blank the root index node's LEB: the strict open now fails
        // exactly like the broken captures do
        let idx = image.master.root_lnum as usize;
        let mut broken = image.lebs.clone();
        broken[idx] = None;
        assert!(
            crate::reader::Ubifs::new(broken.clone()).is_err(),
            "the strict open must refuse a broken index"
        );

        // the rebuild ignores the index and recovers the whole tree
        let (plan, rep) = rebuild(&broken).unwrap();
        assert_eq!(rep.dropped_unreachable_inos, 0);
        assert!(rep.dropped_zero_nlink_files.is_empty());
        assert_eq!(rep.stale_blocks_dropped, 0);
        let mut tree = plan.root;
        let mut expect = crate::reader::Ubifs::new(image.lebs.clone())
            .unwrap()
            .extract()
            .unwrap();
        canonical(&mut tree);
        canonical(&mut expect);
        assert_eq!(fingerprint(&tree), fingerprint(&expect));
    }

    #[test]
    fn masterless_volume_still_rebuilds() {
        let image = mkfs::mkfs(&sample_tree(), &geom_options()).unwrap();
        let mut broken = image.lebs.clone();
        broken[UBIFS_MST_LNUM as usize] = None;
        broken[UBIFS_MST_LNUM as usize + 1] = None;
        assert!(crate::reader::Ubifs::new(broken.clone()).is_err());
        // no committed-state boundary is consulted — the official
        // rebuild does not need the master either
        let (mut plan, rep) = rebuild(&broken).unwrap();
        assert!(!rep.root_created);
        canonical(&mut plan.root);
        let mut expect = crate::reader::Ubifs::new(image.lebs.clone())
            .unwrap()
            .extract()
            .unwrap();
        canonical(&mut expect);
        assert_eq!(fingerprint(&plan.root), fingerprint(&expect));
    }

    #[test]
    fn lost_inode_drops_its_dent() {
        let image = mkfs::mkfs(&sample_tree(), &geom_options()).unwrap();
        let target = dent_target(&image, "marker");
        let (ileb, ipos, ilen) = node_span(&image, |node, ch| {
            ch.node_type == UBIFS_INO_NODE
                && InoNode::parse(node).map(|i| i.key.inum == target) == Ok(true)
        })
        .expect("marker inode node");
        let broken = excise(&image, ileb, ipos, ilen);
        assert!(crate::reader::Ubifs::new(broken.clone()).is_err());
        let (plan, rep) = rebuild(&broken).unwrap();
        assert_eq!(rep.dropped_dangling_dents, 1, "report: {rep:?}");
        assert_eq!(rep.dropped_unreachable_inos, 0);
        // marker is gone, everything else is in place
        if let Kind::Dir(children) = &plan.root.kind {
            let etc = children.iter().find(|(n, _)| n == "etc").unwrap();
            if let Kind::Dir(etc) = &etc.1.kind {
                assert!(!etc.iter().any(|(n, _)| n == "marker"));
                assert!(etc.iter().any(|(n, _)| n == "big"));
            }
            assert!(children.iter().any(|(n, _)| n == "home"));
        }
        // the rebuilt image is a valid filesystem again
        let image2 = mkfs::mkfs_plan(&plan, &geom_options()).unwrap();
        assert!(crate::reader::Ubifs::new(image2.lebs.clone()).is_ok());
    }

    #[test]
    fn hole_is_zero_filled_never_truncated() {
        let image = mkfs::mkfs(&sample_tree(), &geom_options()).unwrap();
        let big = dent_target(&image, "big");
        // excise a middle block: the strict open refuses, the rebuild
        // must keep the size, zero the hole and keep the tail
        let (dleb, dpos, dlen) = data_span(&image, big, 5).expect("block 5");
        let broken = excise(&image, dleb, dpos, dlen);
        assert!(crate::reader::Ubifs::new(broken.clone()).is_err());
        let (plan, rep) = rebuild(&broken).unwrap();
        assert_eq!(rep.zero_filled_blocks, 1);
        assert_eq!(rep.stale_blocks_dropped, 0);
        let mut tree = plan.root;
        let file = child_mut(child_mut(&mut tree, "etc"), "big");
        let Kind::File(content) = &file.kind else {
            panic!("not a file")
        };
        assert_eq!(content.len(), 400_000, "the size is kept");
        assert!(content[5 * 4096..6 * 4096].iter().all(|&b| b == 0), "hole");
        // the tail blocks survived intact
        for (i, b) in content[10 * 4096..11 * 4096].iter().enumerate() {
            let off = 10 * 4096 + i;
            assert_eq!(*b, (off % 251) as u8, "tail byte {off}");
        }
    }

    #[test]
    fn missing_first_block_keeps_the_file() {
        let image = mkfs::mkfs(&sample_tree(), &geom_options()).unwrap();
        let big = dent_target(&image, "big");
        let (dleb, dpos, dlen) = data_span(&image, big, 0).expect("block 0");
        let broken = excise(&image, dleb, dpos, dlen);
        assert!(crate::reader::Ubifs::new(broken.clone()).is_err());
        let (mut plan, rep) = rebuild(&broken).unwrap();
        assert_eq!(rep.zero_filled_blocks, 1);
        let file = child_mut(child_mut(&mut plan.root, "etc"), "big");
        let Kind::File(content) = &file.kind else {
            panic!("not a file")
        };
        assert_eq!(content.len(), 400_000);
        assert!(content[..4096].iter().all(|&b| b == 0), "zeroed head");
        let off = 4096 + 100;
        assert_eq!(content[off], (off % 251) as u8, "the rest is intact");
    }

    #[test]
    fn truncation_drops_stale_blocks() {
        let image = mkfs::mkfs(&sample_tree(), &geom_options()).unwrap();
        let big = dent_target(&image, "big");
        // a truncation newer than every node: 400000 -> 5000
        let broken = append_node(
            &image.lebs,
            &trun_node(big, 400_000, 5_000, max_sqnum(&image) + 5),
        );
        let (mut plan, rep) = rebuild(&broken).unwrap();
        // 98 blocks total, block 0 (4096 <= 5000) survives, 97 are stale
        assert_eq!(rep.stale_blocks_dropped, 97, "report: {rep:?}");
        let file = child_mut(child_mut(&mut plan.root, "etc"), "big");
        let Kind::File(content) = &file.kind else {
            panic!("not a file")
        };
        assert_eq!(content.len(), 5_000, "size = trun point");
        for (i, b) in content[..4096].iter().enumerate() {
            assert_eq!(*b, (i % 251) as u8);
        }
        assert!(content[4096..].iter().all(|&b| b == 0), "past the trun");
    }

    #[test]
    fn deletion_dent_removes_the_name() {
        let image = mkfs::mkfs(&sample_tree(), &geom_options()).unwrap();
        let root_inum = {
            let (idx, pos, len) = dent_span(&image, "small.txt").unwrap();
            let leb = image.lebs[idx].as_ref().unwrap();
            let (node, _) = parse_node(&leb[pos..pos + len], len, "t").unwrap();
            DentNode::parse(node).unwrap().key.inum
        };
        // a deletion dent (inum 0) newer than the live one
        let broken = append_node(
            &image.lebs,
            &crafted_dent(
                root_inum,
                "small.txt",
                0,
                UBIFS_ITYPE_REG,
                max_sqnum(&image) + 10,
            ),
        );
        let (plan, rep) = rebuild(&broken).unwrap();
        assert_eq!(rep.deleted_names, 1, "report: {rep:?}");
        // the name is gone and the orphaned inode is deleted — no
        // lost+found in the official rebuild
        assert_eq!(rep.dropped_unreachable_inos, 1);
        let Kind::Dir(children) = &plan.root.kind else {
            panic!()
        };
        assert!(!children.iter().any(|(n, _)| n == "small.txt"));
        assert!(children.iter().any(|(n, _)| n == "etc"));
    }

    #[test]
    fn zero_nlink_inode_is_deleted() {
        let image = mkfs::mkfs(&sample_tree(), &geom_options()).unwrap();
        let target = dent_target(&image, "small.txt");
        let root_inum = {
            let (idx, pos, len) = dent_span(&image, "small.txt").unwrap();
            let leb = image.lebs[idx].as_ref().unwrap();
            let (node, _) = parse_node(&leb[pos..pos + len], len, "t").unwrap();
            DentNode::parse(node).unwrap().key.inum
        };
        // the deletion pair of the last unlink: a newer nlink-0 inode
        // and the deletion dent — the file must disappear completely
        let with_ino = append_node(
            &image.lebs,
            &crafted_ino(target, 0o100644, 0, max_sqnum(&image) + 10),
        );
        let broken = append_node(
            &with_ino,
            &crafted_dent(
                root_inum,
                "small.txt",
                0,
                UBIFS_ITYPE_REG,
                max_sqnum(&image) + 11,
            ),
        );
        let (plan, rep) = rebuild(&broken).unwrap();
        assert_eq!(rep.deleted_names, 1);
        // the newest ino (nlink 0) is the deletion marker of the last
        // unlink — it goes silently, it is not an orphaned file
        assert!(rep.dropped_zero_nlink_files.is_empty());
        assert_eq!(rep.dropped_unreachable_inos, 0);
        let Kind::Dir(children) = &plan.root.kind else {
            panic!()
        };
        assert!(!children.iter().any(|(n, _)| n == "small.txt"));
    }

    #[test]
    fn uncommitted_tail_is_recovered() {
        let image = mkfs::mkfs(&sample_tree(), &geom_options()).unwrap();
        // a ghost file written right after the committed prefix with
        // sqnums above the master's — a bud tail after an interrupted
        // commit. The official scan keeps it; so must we.
        let with_ino = append_node(
            &image.lebs,
            &crafted_ino(9999, 0o100644, 1, max_sqnum(&image) + 1000),
        );
        let root_inum = {
            let (idx, pos, len) = dent_span(&image, "small.txt").unwrap();
            let leb = image.lebs[idx].as_ref().unwrap();
            let (node, _) = parse_node(&leb[pos..pos + len], len, "t").unwrap();
            DentNode::parse(node).unwrap().key.inum
        };
        let broken = append_node(
            &with_ino,
            &crafted_dent(
                root_inum,
                "ghost",
                9999,
                UBIFS_ITYPE_REG,
                max_sqnum(&image) + 1001,
            ),
        );
        let (plan, rep) = rebuild(&broken).unwrap();
        assert_eq!(rep.dropped_unreachable_inos, 0, "report: {rep:?}");
        assert!(rep.dropped_zero_nlink_files.is_empty());
        let Kind::Dir(children) = &plan.root.kind else {
            panic!()
        };
        assert!(
            children.iter().any(|(n, _)| n == "ghost"),
            "the uncommitted file is recovered"
        );
        let ghost = children.iter().find(|(n, _)| n == "ghost").unwrap();
        assert!(matches!(&ghost.1.kind, Kind::File(d) if d.is_empty()));
    }

    #[test]
    fn dent_cycle_is_cut_not_looped() {
        let image = mkfs::mkfs(&sample_tree(), &geom_options()).unwrap();
        let root_inum = {
            let (idx, pos, len) = dent_span(&image, "small.txt").unwrap();
            let leb = image.lebs[idx].as_ref().unwrap();
            let (node, _) = parse_node(&leb[pos..pos + len], len, "t").unwrap();
            DentNode::parse(node).unwrap().key.inum
        };
        // g1 under the root, g2 under g1, and g1 under g2 — a cycle
        let with_ino1 = append_node(
            &image.lebs,
            &crafted_ino(9001, 0o040755, 2, max_sqnum(&image) + 10),
        );
        let with_ino2 = append_node(
            &with_ino1,
            &crafted_ino(9002, 0o040755, 2, max_sqnum(&image) + 11),
        );
        let with_d1 = append_node(
            &with_ino2,
            &crafted_dent(
                root_inum,
                "cyc1",
                9001,
                UBIFS_ITYPE_DIR,
                max_sqnum(&image) + 12,
            ),
        );
        let with_d2 = append_node(
            &with_d1,
            &crafted_dent(9001, "inner", 9002, UBIFS_ITYPE_DIR, max_sqnum(&image) + 13),
        );
        let broken = append_node(
            &with_d2,
            &crafted_dent(9002, "back", 9001, UBIFS_ITYPE_DIR, max_sqnum(&image) + 14),
        );
        // must terminate. The single-dent rule keeps only the newest
        // dent of a directory, so the cycle pair loses its root-side
        // dent, becomes unreachable and is deleted — the same outcome
        // as the official rebuild (cycle → DENTRY_IS_UNREACHABLE).
        let (plan, rep) = rebuild(&broken).unwrap();
        assert_eq!(rep.dropped_unreachable_inos, 2, "report: {rep:?}");
        let Kind::Dir(children) = &plan.root.kind else {
            panic!()
        };
        assert!(!children.iter().any(|(n, _)| n == "cyc1"), "cycle gone");
        assert!(
            children.iter().any(|(n, _)| n == "etc"),
            "the rest is intact"
        );
    }

    #[test]
    fn hard_links_survive_as_links() {
        // an image with a real hard link: two dents, one inode
        let mut root = sample_tree();
        if let Kind::Dir(children) = &mut root.kind {
            children.push((
                "again.txt".into(),
                Inode::with_kind(
                    0o100644,
                    Kind::Hardlink {
                        target: vec!["small.txt".into()],
                    },
                ),
            ));
        }
        let plan_in = crate::walk::Plan {
            root,
            link_groups: vec![LinkGroup {
                tree_path: vec!["small.txt".into()],
                nlink: 2,
                order_key: 0,
            }],
        };
        let image = mkfs::mkfs_plan(&plan_in, &geom_options()).unwrap();
        assert!(crate::reader::Ubifs::new(image.lebs.clone()).is_ok());

        let (plan, rep) = rebuild(&image.lebs).unwrap();
        assert_eq!(rep.dropped_unreachable_inos, 0, "report: {rep:?}");
        // one link group with nlink 2; the children are name-sorted,
        // so "again.txt" owns the content and "small.txt" is a marker
        assert_eq!(plan.link_groups.len(), 1);
        assert_eq!(plan.link_groups[0].nlink, 2);
        assert_eq!(plan.link_groups[0].tree_path, vec!["again.txt".to_string()]);
        let Kind::Dir(children) = &plan.root.kind else {
            panic!()
        };
        let again = children.iter().find(|(n, _)| n == "again.txt").unwrap();
        assert!(matches!(&again.1.kind, Kind::File(d) if d == b"hello"));
        assert_eq!(again.1.nlink, 2, "calc_nlink counts the dents");
        let small = children.iter().find(|(n, _)| n == "small.txt").unwrap();
        assert!(
            matches!(&small.1.kind, Kind::Hardlink { target } if target == &vec!["again.txt".to_string()])
        );
        // and the plan round-trips into a valid image
        let image2 = mkfs::mkfs_plan(&plan, &geom_options()).unwrap();
        assert!(crate::reader::Ubifs::new(image2.lebs.clone()).is_ok());
    }

    #[test]
    fn missing_root_creates_an_empty_filesystem() {
        let image = mkfs::mkfs(&sample_tree(), &geom_options()).unwrap();
        let (ileb, ipos, ilen) = node_span(&image, |node, ch| {
            ch.node_type == UBIFS_INO_NODE
                && InoNode::parse(node).map(|i| i.key.inum == UBIFS_ROOT_INO) == Ok(true)
        })
        .expect("root inode node");
        let broken = excise(&image, ileb, ipos, ilen);
        assert!(crate::reader::Ubifs::new(broken.clone()).is_err());
        // the official rebuild never fails on a lost root: everything
        // becomes unreachable and a fresh root is created
        let (plan, rep) = rebuild(&broken).unwrap();
        assert!(rep.root_created, "report: {rep:?}");
        // 8 non-root files in the sample tree
        assert_eq!(rep.dropped_unreachable_inos, 8);
        assert_eq!(plan.link_groups.len(), 0);
        assert!(plan.root.validate().is_ok());
        let image2 = mkfs::mkfs_plan(&plan, &geom_options()).unwrap();
        assert!(crate::reader::Ubifs::new(image2.lebs.clone()).is_ok());
    }
}
