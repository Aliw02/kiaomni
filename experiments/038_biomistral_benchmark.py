"""
038_biomistral_benchmark.py — KV-Cache Eviction on Biomedical LLM
===================================================================
Platform : L4 (24 GB VRAM)
Model    : BioMistral/BioMistral-7B (NF4 4-bit, bfloat16 compute)

Policies : SnapKV_Modified, RealSnapKV, Ada-SnapKV, H2O,
            KiaOmni_s8, KiaOmni_Adaptive, KiaOmni_RatioAdaptive, KiaOmni_Quest,
            KiaOmni_Gaussian, KiaOmni_AnchorExp, KiaOmni_Scissorhands

Bio-RULER Tasks:
  bio_niah_single  — Real dbSNP rsID buried in genomic literature filler
  bio_niah_gene    — Real HGVS gene mutation buried in clinical text filler
  bio_vt           — Drug interaction chain (3 real drug names, multi-hop)

Bio-LB Tasks (real open datasets via HuggingFace):
  pubmedqa, pubmedqa_long, medmcqa, medalpaca_medqa, medalpaca_wiki, clinical_niah

Contexts : {4096, 8192, 16384}
Budgets  : {96, 128, 256, 512}
N        : 15 Bio-RULER trials / 15 Bio-LB samples per cell

Run:
    python experiments/038_biomistral_benchmark.py

Outputs:
    experiments/results/038_biomistral_results/results.json
    experiments/results/038_biomistral_results/predictions.csv
    experiments/results/038_biomistral_results/speed_vram.csv
    experiments/results/038_biomistral_results/eviction_coherence_loss.csv
    experiments/results/038_biomistral_results/checkpoints/
"""

import csv, gc, json, math, os, random, re, string, collections, time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
transformers.logging.set_verbosity_error()

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_NAME   = "BioMistral/BioMistral-7B"
CTX_LENS     = [4096, 8192, 16384]
BUDGETS      = [96, 128, 256, 512]
N_TRIALS     = 15
LB_SAMPLES   = 15
BIO_RULER_TASKS = ["bio_niah_single", "bio_niah_gene", "bio_vt"]
BIO_LB_TASKS    = [
    "pubmedqa",
    "pubmedqa_long",
    "medmcqa",
    "medalpaca_medqa",
    "medalpaca_wiki",
    "clinical_niah",
]
SEED         = 42
MAX_NEW      = 96

N_SINK      = 16
RECENCY     = 32
BLOCK_SIZE  = 16
SIGMA_FIXED = 8
SIGMA_MAX   = 64
SNAP_POOL_K = 5
SNAP_OBS_W  = 32

CHAIN_LEN   = 3

OUT_DIR = Path("results/038_biomistral_results")
CKPT_DIR           = OUT_DIR / "checkpoints"
PRED_CSV_PATH      = OUT_DIR / "predictions.csv"
SPEED_CSV_PATH     = OUT_DIR / "speed_vram.csv"
COHERENCE_CSV_PATH = OUT_DIR / "eviction_coherence_loss.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

RULER_DEPTHS = [0.25, 0.5, 0.75]
METRIC_KEYS  = ["f1", "em", "rouge_l", "contains"]

PRED_COLS = [
    "source", "task", "ctx", "trial_or_sample", "policy", "budget",
    "ground_truth", "prediction", "f1", "em", "rouge_l", "contains",
    "llm_judge_score", "llm_judge_reason",
]
SPEED_COLS = [
    "source", "task", "ctx", "trial_or_sample", "policy", "budget",
    "sal_ms", "gen_ms", "tokens_per_sec", "vram_sal_mb", "vram_gen_mb",
]
COHERENCE_COLS = [
    "source", "task", "ctx", "trial_or_sample", "policy", "budget",
    "eviction_coherence_loss",
]

SNP_POOL = [
    ("rs7412",    "APOE",   "Alzheimer's risk / lipid metabolism"),
    ("rs429358",  "APOE",   "Alzheimer's risk (e4 allele)"),
    ("rs1799945", "HFE",    "hereditary hemochromatosis H63D"),
    ("rs1800562", "HFE",    "hereditary hemochromatosis C282Y"),
    ("rs1801133", "MTHFR",  "folate metabolism C677T"),
    ("rs1801131", "MTHFR",  "folate metabolism A1298C"),
    ("rs1799853", "CYP2C9", "warfarin metabolism *2 variant"),
    ("rs1057910", "CYP2C9", "warfarin metabolism *3 variant"),
    ("rs4244285", "CYP2C19","clopidogrel non-response *2"),
    ("rs4986893", "CYP2C19","clopidogrel non-response *3"),
    ("rs1045642", "ABCB1",  "multidrug resistance transporter"),
    ("rs2032582", "ABCB1",  "P-glycoprotein drug efflux"),
    ("rs9923231", "VKORC1", "warfarin sensitivity -1639G>A"),
    ("rs2108622", "CYP4F2", "warfarin dose vitamin K"),
    ("rs4149056", "SLCO1B1","statin-induced myopathy"),
    ("rs776746",  "CYP3A5", "tacrolimus metabolism *3"),
    ("rs3892097", "CYP2D6", "codeine/tramadol *4 poor metabolizer"),
    ("rs5030655", "CYP2D6", "reduced activity *6"),
    ("rs28371725","CYP2D6", "reduced activity *41"),
    ("rs1799930", "NAT2",   "isoniazid slow acetylator"),
    ("rs1208",    "NAT2",   "caffeine / drug acetylation"),
    ("rs1801280", "NAT2",   "slow acetylator phenotype"),
    ("rs4680",    "COMT",   "pain sensitivity / dopamine Val158Met"),
    ("rs6311",    "HTR2A",  "antidepressant response serotonin"),
    ("rs1800497", "ANKK1",  "DRD2 Taq1A antipsychotic response"),
    ("rs4633",    "COMT",   "neuropsychiatric disorder risk"),
    ("rs334",     "HBB",    "sickle cell disease Val6Glu"),
    ("rs11549407","HBB",    "hemoglobin variant"),
    ("rs1800465", "TPMT",   "thiopurine methyltransferase *2"),
    ("rs1142345", "TPMT",   "thiopurine *3C azathioprine toxicity"),
]

GENE_MUTATION_POOL = [
    ("BRCA1", "c.5266dupC",    "p.Gln1756Profs*74", "hereditary breast and ovarian cancer"),
    ("BRCA2", "c.6174delT",    "p.Ser2058Argfs*7",  "Ashkenazi Jewish founder mutation breast cancer"),
    ("CFTR",  "c.1521_1523delCTT", "p.Phe508del",   "cystic fibrosis most common variant"),
    ("TP53",  "c.817C>T",      "p.Arg273Cys",       "Li-Fraumeni syndrome somatic mutation"),
    ("KRAS",  "c.35G>T",       "p.Gly12Val",        "colorectal cancer activating oncogene"),
    ("EGFR",  "c.2573T>G",     "p.Leu858Arg",       "non-small cell lung cancer EGFR inhibitor sensitivity"),
    ("BRAF",  "c.1799T>A",     "p.Val600Glu",       "melanoma BRAF V600E targeted therapy"),
    ("MLH1",  "c.1852_1853delinsTG", "p.Lys618*",   "Lynch syndrome colorectal cancer MMR deficiency"),
    ("APC",   "c.3927_3931delAAAGA", "p.Glu1309Aspfs*4", "familial adenomatous polyposis"),
    ("PTEN",  "c.697C>T",      "p.Arg233*",         "Cowden syndrome PTEN hamartoma tumor syndrome"),
    ("VHL",   "c.482G>A",      "p.Arg161Gln",       "von Hippel-Lindau disease renal cell carcinoma"),
    ("RET",   "c.1900T>C",     "p.Cys634Arg",       "multiple endocrine neoplasia type 2A"),
    ("MEN1",  "c.1378C>T",     "p.Arg460*",         "multiple endocrine neoplasia type 1"),
    ("NF1",   "c.2041C>T",     "p.Arg681*",         "neurofibromatosis type 1"),
    ("NF2",   "c.880C>T",      "p.Arg294*",         "neurofibromatosis type 2 bilateral schwannomas"),
    ("ATM",   "c.7271T>G",     "p.Val2424Gly",      "ataxia-telangiectasia breast cancer risk"),
    ("CHEK2", "c.1100delC",    "p.Thr367Metfs*15",  "hereditary breast cancer moderate risk"),
    ("PALB2", "c.3113G>A",     "p.Trp1038*",        "PALB2-associated breast cancer"),
    ("RAD51C","c.576+1G>A",    "p.?",               "Fanconi anemia BRCA-related ovarian cancer"),
    ("MUTYH", "c.536A>G",      "p.Tyr179Cys",       "MUTYH-associated polyposis colorectal cancer"),
]

DRUG_INTERACTION_PAIRS = [
    ("warfarin",     "amiodarone",   "increased anticoagulation INR elevation"),
    ("methotrexate", "NSAIDs",       "reduced renal clearance toxicity risk"),
    ("digoxin",      "quinidine",    "elevated digoxin plasma level arrhythmia"),
    ("simvastatin",  "clarithromycin","rhabdomyolysis CYP3A4 inhibition"),
    ("clopidogrel",  "omeprazole",   "reduced antiplatelet efficacy CYP2C19"),
    ("tacrolimus",   "fluconazole",  "calcineurin inhibitor toxicity CYP3A4"),
    ("lithium",      "thiazide diuretics", "lithium toxicity renal reabsorption"),
    ("carbamazepine","valproate",    "reduced carbamazepine epoxide metabolism"),
    ("phenytoin",    "fluoxetine",   "phenytoin toxicity CYP2C9 inhibition"),
    ("theophylline", "ciprofloxacin","theophylline toxicity CYP1A2 inhibition"),
    ("rifampicin",   "oral contraceptives", "contraceptive failure CYP induction"),
    ("MAOIs",        "SSRIs",        "serotonin syndrome life-threatening interaction"),
    ("tramadol",     "MAOIs",        "seizure serotonin syndrome opioid interaction"),
    ("azathioprine", "allopurinol",  "6-mercaptopurine toxicity TPMT inhibition"),
    ("metformin",    "contrast media","lactic acidosis renal impairment"),
]

ICD_POOL = [
    ("I21.0", "ST elevation myocardial infarction anterior wall"),
    ("J18.9", "pneumonia unspecified organism"),
    ("E11.9", "type 2 diabetes mellitus without complications"),
    ("I63.9", "cerebral infarction unspecified"),
    ("C34.10","malignant neoplasm of upper lobe bronchus or lung"),
    ("M54.5", "low back pain"),
    ("F32.1", "major depressive disorder single episode moderate"),
    ("K92.1", "melena lower gastrointestinal bleeding"),
    ("N18.3", "chronic kidney disease stage 3"),
    ("G35",   "multiple sclerosis"),
    ("M05.79","rheumatoid arthritis multiple sites seronegative"),
    ("C50.911","malignant neoplasm breast upper outer quadrant"),
    ("I48.91","unspecified atrial fibrillation"),
    ("J44.1", "chronic obstructive pulmonary disease acute exacerbation"),
    ("Z79.01","long-term current use of anticoagulants"),
]

GENOMIC_FILLER = [
    "Genome-wide association studies have identified thousands of loci associated with complex human diseases.",
    "Next-generation sequencing technologies have revolutionized the identification of pathogenic variants.",
    "The human genome contains approximately 20,000 protein-coding genes distributed across 23 chromosome pairs.",
    "Single nucleotide polymorphisms represent the most common form of genetic variation in the human genome.",
    "Copy number variants involving large genomic segments contribute significantly to phenotypic diversity.",
    "Chromatin remodeling complexes regulate gene expression through epigenetic modification of histone proteins.",
    "RNA splicing variants can alter protein function through disruption of canonical splice donor or acceptor sites.",
    "Pharmacogenomic testing enables personalized drug selection based on an individual's genetic profile.",
    "Somatic mutations accumulate in cancer cells due to defective DNA repair mechanisms and replication errors.",
    "Germline pathogenic variants in mismatch repair genes predispose individuals to Lynch syndrome cancers.",
    "Loss of heterozygosity at tumor suppressor gene loci is a common event in malignant transformation.",
    "Mitochondrial DNA mutations are maternally inherited and can cause multisystem metabolic disorders.",
    "Trinucleotide repeat expansions underlie several neurodegenerative disorders including Huntington disease.",
    "CRISPR-Cas9 genome editing enables precise correction of pathogenic variants in patient-derived cells.",
    "Polygenic risk scores aggregate the effects of many common variants to predict disease susceptibility.",
    "The allele frequency of a variant in population databases helps distinguish pathogenic from benign variants.",
    "Protein truncating variants in tumor suppressor genes are typically classified as likely pathogenic.",
    "Missense variants require functional evidence to determine their clinical significance for disease causation.",
    "Variant interpretation follows ACMG-AMP guidelines combining population, computational and functional data.",
    "Structural genomic variants including inversions and translocations can disrupt gene regulatory elements.",
    "Epigenome-wide association studies examine DNA methylation patterns in relation to disease phenotypes.",
    "Long-read sequencing technologies improve phasing of variants and resolution of complex genomic regions.",
    "Pharmacokinetic gene variants affect drug absorption, distribution, metabolism and excretion properties.",
    "Founder variants in specific population groups reach elevated frequency through genetic drift and bottlenecks.",
    "Phenotypic penetrance describes the proportion of individuals with a genotype who manifest the associated phenotype.",
    "Variable expressivity refers to differences in phenotypic severity among individuals with the same genotype.",
    "Consanguinity increases the probability of autosomal recessive conditions through shared ancestral haplotypes.",
    "The gnomAD database provides population-level variant frequency data from over 125,000 exome sequences.",
    "ClinVar aggregates clinician-submitted variant classifications linked to phenotypic and functional evidence.",
    "OMIM catalogs the genetic basis of inherited disorders and maps genotype-phenotype correlations.",
    "Familial segregation analysis provides evidence for variant pathogenicity in Mendelian disease families.",
    "Multigene panel testing improves diagnostic yield compared to single-gene testing in hereditary cancer.",
    "Chromosomal microarray detects copy number variants too small to be identified by conventional karyotyping.",
    "Whole exome sequencing captures all protein-coding regions and has transformed rare disease diagnosis.",
    "Whole genome sequencing additionally captures non-coding regulatory regions and deep intronic variants.",
    "RNA sequencing can detect cryptic splice variants that escape detection by DNA sequencing alone.",
    "The Human Phenotype Ontology provides standardized terms for describing phenotypic abnormalities in patients.",
    "Cascade genetic testing of relatives enables identification of at-risk family members for preventive care.",
    "Preimplantation genetic testing allows selection of unaffected embryos during assisted reproduction.",
    "Somatic variant allele frequency in tumor samples reflects the clonal architecture of the malignancy.",
]

CLINICAL_FILLER = [
    "Electronic health records contain longitudinal clinical data enabling retrospective cohort studies.",
    "Clinical decision support systems alert prescribers to potential drug-drug interactions at the point of care.",
    "Randomized controlled trials represent the gold standard for evaluating therapeutic interventions.",
    "Systematic reviews and meta-analyses synthesize evidence from multiple studies to inform clinical guidelines.",
    "Biomarker-driven patient stratification enables precision medicine approaches in oncology.",
    "Adverse drug reactions represent a significant cause of preventable morbidity and healthcare costs.",
    "Real-world evidence from claims databases complements clinical trial data in post-marketing surveillance.",
    "Clinical laboratory reference ranges are established from healthy population samples using statistical methods.",
    "Point-of-care testing improves turnaround time for critical laboratory results in emergency settings.",
    "Multidisciplinary tumor boards coordinate specialist input for complex oncology treatment decisions.",
    "Patient-reported outcome measures capture health-related quality of life beyond traditional clinical endpoints.",
    "Artificial intelligence algorithms applied to medical imaging improve detection sensitivity and specificity.",
    "Telemedicine platforms expand access to specialty care for patients in geographically remote locations.",
    "Clinical pathway standardization reduces variation in care delivery and improves patient safety outcomes.",
    "Antimicrobial stewardship programs optimize antibiotic use to reduce resistance development.",
    "Population health management approaches identify high-risk patients for targeted preventive interventions.",
    "Shared decision-making tools help patients understand risks and benefits of treatment options.",
    "Hospital-acquired infections represent a major source of preventable morbidity in healthcare settings.",
    "Physiologically-based pharmacokinetic modeling predicts drug exposure in special populations.",
    "Therapeutic drug monitoring guides individualized dosing of medications with narrow therapeutic indices.",
]

def _build_bio_haystack(rng: random.Random, target_chars: int, bio_type: str = "genomic") -> str:
    filler = GENOMIC_FILLER if bio_type == "genomic" else CLINICAL_FILLER
    sentences = filler[:]
    hay = ""
    while len(hay) < target_chars:
        rng.shuffle(sentences)
        hay += " ".join(sentences) + " "
    return hay[:target_chars]


def load_model():
    print(f"Loading {MODEL_NAME} (4-bit NF4, bfloat16)...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=cfg,
        trust_remote_code=False,
        attn_implementation="eager",
    )
    print("  attn_implementation=eager", flush=True)
    model.eval()
    print("Model ready.", flush=True)
    return model, tok


def extract_all_saliency(ids: torch.Tensor, model) -> dict:
    L_seq = ids.shape[1]
    c     = model.config
    nh    = c.num_attention_heads
    nk    = getattr(c, "num_key_value_heads", nh)
    hd    = c.hidden_size // nh
    n_layers = len(model.model.layers)

    last_k_buf: dict = {}
    sal_per_layer_list: list = [None] * n_layers
    hooks = []

    for l_idx, layer in enumerate(model.model.layers):
        def _make_k_hook(layer_idx, is_last, q_store):
            def _h(m, inp, out):
                k_raw = out.detach().cpu().to(torch.float32)
                q_raw = q_store.get("q")
                if q_raw is None:
                    return
                _L   = ids.shape[1]
                _nh2 = c.num_attention_heads
                _nk2 = getattr(c, "num_key_value_heads", _nh2)
                _hd2 = c.hidden_size // _nh2
                q2 = q_raw.view(1, _L, _nh2, _hd2).transpose(1, 2)
                k2 = k_raw.view(1, _L, _nk2, _hd2).transpose(1, 2)
                if _nk2 != _nh2:
                    k2 = k2.repeat_interleave(_nh2 // _nk2, dim=1)
                sc2 = torch.matmul(q2[:, :, -1:, :], k2.transpose(-2, -1)) * (_hd2 ** -0.5)
                sal_heads = torch.softmax(sc2, dim=-1)[0, :, 0, :]
                sal_mean_l = sal_heads.mean(0).numpy()
                sal_per_layer_list[layer_idx] = sal_mean_l.astype(np.float32)
                if is_last:
                    last_k_buf["sal_heads"] = sal_heads.numpy()
                    obs_w = min(SNAP_OBS_W, _L)
                    q_obs = q2[:, :, -obs_w:, :]
                    sc_obs = torch.matmul(q_obs, k2.transpose(-2, -1)) * (_hd2 ** -0.5)
                    prefix_len = max(1, _L - obs_w)
                    attn_prefix = torch.softmax(sc_obs[..., :prefix_len], dim=-1)
                    votes = attn_prefix.sum(dim=-2)
                    max_v = votes.max(dim=-1, keepdim=True).values
                    pad   = max_v.expand(1, _nh2, obs_w)
                    sal_snapkv_h = torch.cat([votes, pad], dim=-1)[0]
                    last_k_buf["sal_snapkv"] = sal_snapkv_h.numpy()
                    del q_obs, sc_obs, attn_prefix, votes, max_v, pad, sal_snapkv_h
                del q2, k2, sc2, sal_heads
            return _h

        is_last = (l_idx == n_layers - 1)
        _temp_q_store: dict = {}

        def _q_capture(m, inp, out, _store=_temp_q_store):
            _store["q"] = out.detach().cpu().to(torch.float32)

        hooks.append(layer.self_attn.q_proj.register_forward_hook(_q_capture))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(
            _make_k_hook(l_idx, is_last, _temp_q_store)))

    try:
        with torch.no_grad():
            model(ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    sal_heads_last  = last_k_buf.get("sal_heads")
    sal_snapkv_last = last_k_buf.get("sal_snapkv")

    if sal_heads_last is None:
        sal_mean = _last_layer_saliency(ids, model)
        sal_per_head = np.tile(sal_mean, (nh, 1))
    else:
        sal_per_head = sal_heads_last.astype(np.float32)
        sal_mean     = sal_per_head.mean(0)

    sal_snapkv = (
        sal_snapkv_last.astype(np.float32)
        if sal_snapkv_last is not None
        else sal_per_head
    )

    sal_per_layer = np.stack(
        [(x if x is not None else sal_mean) for x in sal_per_layer_list], axis=0
    ).astype(np.float32)

    n_lay = sal_per_layer.shape[0]
    sal_scissor = (
        sal_per_layer[n_lay // 4] +
        sal_per_layer[n_lay // 2] +
        sal_per_layer[-1]
    ) / 3.0

    del last_k_buf
    return {
        "sal_mean":      sal_mean,
        "sal_snapkv":    sal_snapkv,
        "sal_per_head":  sal_per_head,
        "sal_per_layer": sal_per_layer,
        "sal_scissor":   sal_scissor,
    }


def _last_layer_saliency(ids: torch.Tensor, model) -> np.ndarray:
    buf: dict = {}
    last = model.model.layers[-1].self_attn
    h_q = last.q_proj.register_forward_hook(
        lambda *a: buf.update({"q": a[2].detach().cpu().to(torch.float32)}))
    h_k = last.k_proj.register_forward_hook(
        lambda *a: buf.update({"k": a[2].detach().cpu().to(torch.float32)}))
    try:
        with torch.no_grad():
            model(ids, use_cache=False)
    finally:
        h_q.remove(); h_k.remove()
    L = ids.shape[1]; c = model.config
    nh = c.num_attention_heads
    nk = getattr(c, "num_key_value_heads", nh)
    hd = c.hidden_size // nh
    q, k = buf["q"], buf["k"]
    q = q.view(1, L, nh, hd).transpose(1, 2)
    k = k.view(1, L, nk, hd).transpose(1, 2)
    if nk != nh:
        k = k.repeat_interleave(nh // nk, dim=1)
    sc  = torch.matmul(q[:, :, -1:, :], k.transpose(-2, -1)) * (hd ** -0.5)
    sal = torch.softmax(sc, dim=-1)[0, :, 0, :].mean(0).numpy()
    del q, k, sc
    return sal


def _protected(n: int) -> set:
    return set(range(min(N_SINK, n))) | set(range(max(0, n - RECENCY), n))

def _boxcar(x: np.ndarray, sigma: int) -> np.ndarray:
    if sigma <= 0:
        return x.astype(np.float32)
    ps = np.concatenate([[0.0], np.cumsum(x.astype(np.float64))])
    lo = np.maximum(0, np.arange(len(x)) - sigma)
    hi = np.minimum(len(x), np.arange(len(x)) + sigma + 1)
    return ((ps[hi] - ps[lo]) / (hi - lo)).astype(np.float32)


def snapkv_modified_keep(sal: np.ndarray, budget: int, seq_len: int) -> set:
    prot_mask = np.zeros(seq_len, dtype=bool)
    prot_mask[:N_SINK] = True
    prot_mask[max(0, seq_len - RECENCY):] = True
    evict_idx = np.where(~prot_mask)[0]
    if len(evict_idx) == 0 or budget >= seq_len:
        return set(range(seq_len))
    page_ids     = evict_idx // BLOCK_SIZE
    sal_evict    = sal[evict_idx]
    unique_pages = np.unique(page_ids)
    page_scores  = np.array(
        [sal_evict[page_ids == pg].mean() for pg in unique_pages], dtype=np.float32)
    order        = np.argsort(page_scores)
    evicted_mask = np.zeros(seq_len, dtype=bool)
    tokens_evicted = 0
    target_evict = max(0, seq_len - budget)
    for pi in order:
        if tokens_evicted >= target_evict:
            break
        pg_mask = page_ids == unique_pages[pi]
        evicted_mask[evict_idx[pg_mask]] = True
        tokens_evicted += int(pg_mask.sum())
    return set(np.where(~evicted_mask)[0].tolist())


def snapkv_real_keep(sal: np.ndarray, budget: int, seq_len: int) -> set:
    from scipy.ndimage import maximum_filter1d
    if budget >= seq_len:
        return set(range(seq_len))
    prot = _protected(seq_len)
    eff  = max(0, budget - len(prot))
    if eff <= 0:
        return prot
    if sal.ndim == 1:
        sal = sal[np.newaxis, :]
    n_heads  = sal.shape[0]
    k_per_h  = max(1, eff // n_heads)
    free     = np.array([i for i in range(seq_len) if i not in prot])
    if len(free) == 0:
        return prot
    kept: set = set(prot)
    for h in range(n_heads):
        pooled = maximum_filter1d(sal[h, :seq_len].astype(np.float32), size=SNAP_POOL_K)
        k      = min(k_per_h, len(free))
        top    = np.argpartition(pooled[free], -k)[-k:]
        kept  |= set(free[top].tolist())
    if len(kept) > budget:
        mean_sal = sal.mean(0)
        kept_arr = np.array(sorted(kept))
        trim_k   = np.argpartition(mean_sal[kept_arr], -budget)[-budget:]
        kept     = set(kept_arr[trim_k].tolist())
    return kept


def h2o_keep(sal: np.ndarray, budget: int, seq_len: int) -> set:
    prot  = _protected(seq_len)
    free  = max(0, budget - len(prot))
    cands = np.array([i for i in range(seq_len) if i not in prot])
    if free <= 0 or len(cands) == 0:
        return prot
    top = np.argpartition(-sal[cands], min(free, len(cands)) - 1)[:free]
    return set(cands[top].tolist()) | prot


def ada_snapkv_keep(sal_mean, budget, seq_len, n_sink=N_SINK, recency=RECENCY, obs_window=64):
    start = max(0, seq_len - recency - obs_window)
    end   = seq_len - recency
    obs   = sal_mean[start:end]
    if len(obs) == 0:
        ratio = 0
    else:
        p = obs / (obs.sum() + 1e-9)
        H = -(p * np.log(p + 1e-9)).sum()
        H_max = np.log(obs_window + 1e-9)
        ratio = H / H_max
    dynamic_budget = int(budget * (1.0 + 0.5 * ratio))
    dynamic_budget = min(dynamic_budget, seq_len)
    protected = _protected(seq_len)
    eff = max(0, dynamic_budget - len(protected))
    sal = sal_mean.copy()
    sal[list(protected)] = -np.inf
    topk = np.argpartition(sal, -eff)[-eff:] if eff > 0 else np.array([], dtype=int)
    return set(topk.tolist()) | protected


def get_adaptive_sigma(sal: np.ndarray, budget: int, seq_len: int) -> int:
    p = sal / (np.sum(sal) + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))
    h_norm = entropy / np.log(max(seq_len, 2))
    peakiness = max(0.0, 1.0 - h_norm)
    return int(max(1, round(SIGMA_MAX * peakiness * np.sqrt(budget / seq_len))))

def get_ratio_adaptive_sigma(sal: np.ndarray, budget: int, seq_len: int) -> int:
    p = sal / (sal.sum() + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))
    h_norm = entropy / np.log(max(seq_len, 2))
    peakiness = max(0.0, 1.0 - h_norm)
    return int(max(1, round((seq_len / max(1, budget)) * peakiness)))

def _quest_envelope(x: np.ndarray, sigma: int) -> np.ndarray:
    if sigma <= 0:
        return x.astype(np.float32)
    from scipy.ndimage import maximum_filter1d
    return maximum_filter1d(x.astype(np.float32), size=(2 * sigma) + 1)

def _gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x.astype(np.float32)
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(x.astype(np.float32), sigma=sigma)


def kiaomni_fixed_keep(sal: np.ndarray, budget: int, seq_len: int, sigma: int = SIGMA_FIXED) -> set:
    prot  = _protected(seq_len)
    F     = _boxcar(np.log1p(sal), sigma)
    free  = max(0, budget - len(prot))
    cands = np.array([i for i in range(seq_len) if i not in prot])
    if free <= 0 or len(cands) == 0:
        return prot
    top = np.argpartition(-F[cands], min(free, len(cands)) - 1)[:free]
    return set(cands[top].tolist()) | prot

def kiaomni_adaptive_keep(sal: np.ndarray, budget: int, seq_len: int) -> set:
    return kiaomni_fixed_keep(sal, budget, seq_len, get_adaptive_sigma(sal, budget, seq_len))

def kiaomni_ratio_adaptive_keep(sal: np.ndarray, budget: int, seq_len: int) -> set:
    return kiaomni_fixed_keep(sal, budget, seq_len, get_ratio_adaptive_sigma(sal, budget, seq_len))

def kiaomni_quest_keep(sal: np.ndarray, budget: int, seq_len: int, sigma: int = SIGMA_FIXED) -> set:
    prot  = _protected(seq_len)
    F     = _quest_envelope(np.log1p(sal), sigma)
    free  = max(0, budget - len(prot))
    cands = np.array([i for i in range(seq_len) if i not in prot])
    if free <= 0 or len(cands) == 0:
        return prot
    top = np.argpartition(-F[cands], min(free, len(cands)) - 1)[:free]
    return set(cands[top].tolist()) | prot

def kiaomni_gaussian_keep(sal: np.ndarray, budget: int, seq_len: int, sigma: float = 4.0) -> set:
    prot  = _protected(seq_len)
    F     = _gaussian_smooth(np.log1p(sal), sigma)
    free  = max(0, budget - len(prot))
    cands = np.array([i for i in range(seq_len) if i not in prot])
    if free <= 0 or len(cands) == 0:
        return prot
    top = np.argpartition(-F[cands], min(free, len(cands)) - 1)[:free]
    return set(cands[top].tolist()) | prot

def kiaomni_anchor_expand_keep(sal: np.ndarray, budget: int, seq_len: int, radius: int = 5) -> set:
    prot = _protected(seq_len)
    free = max(0, budget - len(prot))
    if free <= 0:
        return prot
    cands = np.array([i for i in range(seq_len) if i not in prot])
    if len(cands) == 0:
        return prot
    sorted_idx = cands[np.argsort(-sal[cands])]
    keep_set = set(prot)
    for anchor in sorted_idx:
        if len(keep_set) >= budget:
            break
        for j in range(max(0, anchor - radius), min(seq_len, anchor + radius + 1)):
            if len(keep_set) >= budget:
                break
            keep_set.add(j)
    return keep_set

def kiaomni_scissorhands_keep(sal_scissor: np.ndarray, budget: int, seq_len: int) -> set:
    return kiaomni_fixed_keep(sal_scissor, budget, seq_len, sigma=0)


POLICIES: dict = {
    "SnapKV_Modified":       ("sal_mean",    lambda s, B, L: snapkv_modified_keep(s["sal_mean"], B, L)),
    "RealSnapKV":            ("sal_snapkv",  lambda s, B, L: snapkv_real_keep(s["sal_snapkv"], B, L)),
    "Ada-SnapKV":            ("sal_mean",    lambda s, B, L: ada_snapkv_keep(s["sal_mean"], B, L)),
    "H2O":                   ("sal_mean",    lambda s, B, L: h2o_keep(s["sal_mean"], B, L)),
    "KiaOmni_s8":            ("sal_mean",    lambda s, B, L: kiaomni_fixed_keep(s["sal_mean"], B, L)),
    "KiaOmni_Adaptive":      ("sal_mean",    lambda s, B, L: kiaomni_adaptive_keep(s["sal_mean"], B, L)),
    "KiaOmni_RatioAdaptive": ("sal_mean",    lambda s, B, L: kiaomni_ratio_adaptive_keep(s["sal_mean"], B, L)),
    "KiaOmni_Quest":         ("sal_mean",    lambda s, B, L: kiaomni_quest_keep(s["sal_mean"], B, L)),
    "KiaOmni_Gaussian":      ("sal_mean",    lambda s, B, L: kiaomni_gaussian_keep(s["sal_mean"], B, L)),
    "KiaOmni_AnchorExp":     ("sal_mean",    lambda s, B, L: kiaomni_anchor_expand_keep(s["sal_mean"], B, L)),
    "KiaOmni_Scissorhands":  ("sal_scissor", lambda s, B, L: kiaomni_scissorhands_keep(s["sal_scissor"], B, L)),
}


def _fresh_cache():
    try:
        from transformers import DynamicCache
        return DynamicCache()
    except Exception:
        return None

@torch.no_grad()
def gen_evict(model, tok, ids: torch.Tensor, keep: set, max_new: int = MAX_NEW) -> str:
    keep_t = torch.tensor(sorted(keep), device=ids.device, dtype=torch.long)
    p      = ids[:, keep_t]
    cache  = _fresh_cache()
    kwargs = dict(attention_mask=torch.ones_like(p), max_new_tokens=max_new,
                  do_sample=False, pad_token_id=tok.eos_token_id)
    if cache is not None:
        kwargs["past_key_values"] = cache
    out = model.generate(p, **kwargs)
    return tok.decode(out[0, p.shape[1]:], skip_special_tokens=True)

@torch.no_grad()
def gen_full(model, tok, ids: torch.Tensor, max_new: int = MAX_NEW) -> str:
    cache  = _fresh_cache()
    kwargs = dict(attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                  do_sample=False, pad_token_id=tok.eos_token_id)
    if cache is not None:
        kwargs["past_key_values"] = cache
    out = model.generate(ids, **kwargs)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())

def _token_f1(pred: str, truth: str) -> float:
    p, t = _norm(pred).split(), _norm(truth).split()
    if not p or not t:
        return float(p == t)
    common = sum((collections.Counter(p) & collections.Counter(t)).values())
    return 0.0 if common == 0 else 2 * common / (len(p) + len(t))

def _rouge_l(pred: str, truth: str) -> float:
    p, t = _norm(pred).split(), _norm(truth).split()
    if not p or not t:
        return 0.0
    m, n = len(p), len(t)
    dp   = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if p[i-1] == t[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / m, lcs / n
    return 2 * prec * rec / (prec + rec)

try:
    from rouge_score import rouge_scorer as _rs
    _rouge_lib = _rs.RougeScorer(["rougeL"], use_stemmer=True)
    def _rouge_l(pred: str, truth: str) -> float:
        return _rouge_lib.score(_norm(truth), _norm(pred))["rougeL"].fmeasure
except ImportError:
    pass

def compute_metrics(pred: str, ground_truth) -> dict:
    answers = ground_truth if isinstance(ground_truth, list) else [str(ground_truth)]
    best: dict = {"f1": 0.0, "em": 0.0, "rouge_l": 0.0, "contains": 0.0}
    pn = _norm(pred)
    for a in answers:
        an = _norm(str(a))
        best["f1"]       = max(best["f1"],      _token_f1(pred, a))
        best["em"]       = max(best["em"],       float(pn == an))
        best["rouge_l"]  = max(best["rouge_l"],  _rouge_l(pred, a))
        best["contains"] = max(best["contains"], float(an in pn))
    return best


@torch.no_grad()
def measure_coherence_loss(model, ids: torch.Tensor, keep: set) -> float:
    if not keep:
        return float("inf")
    keep_t    = torch.tensor(sorted(keep), device=ids.device, dtype=torch.long)
    ids_evict = ids[:, keep_t]
    if ids_evict.shape[1] < 2:
        return float("inf")
    try:
        out = model(ids_evict, labels=ids_evict)
        return float(torch.exp(out.loss).item())
    except Exception:
        return float("nan")

def _vram_reset() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def _vram_peak_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def build_bio_niah_single(rng: random.Random, tok, target_toks: int, depth: float):
    snp_id, gene, phenotype = rng.choice(SNP_POOL)
    needle = (f"Whole-genome sequencing confirmed the presence of variant {snp_id} "
              f"in the {gene} gene, which is associated with {phenotype}.")
    q_text = "\n\nWhat dbSNP variant identifier was confirmed by whole-genome sequencing? Answer with only the rsID."
    pre    = "Review the following genomic research summary.\n\n"
    hay    = _build_bio_haystack(rng, target_toks * 5, "genomic")
    for _ in range(20):
        split = int(len(hay) * depth)
        full  = pre + hay[:split] + "\n\n" + needle + "\n\n" + hay[split:] + q_text
        ids   = tok(full, return_tensors="pt").input_ids
        if ids.shape[1] <= target_toks + 200:
            break
        hay = hay[:int(len(hay) * 0.92)]
    return ids, snp_id


def build_bio_niah_gene(rng: random.Random, tok, target_toks: int, depth: float):
    gene, cdna, protein, phenotype = rng.choice(GENE_MUTATION_POOL)
    needle = (f"Germline sequencing identified a pathogenic variant {gene} {cdna} "
              f"({protein}), consistent with a diagnosis of {phenotype}.")
    answer = f"{gene} {cdna}"
    q_text = "\n\nWhat pathogenic gene variant was identified by germline sequencing? Provide the gene name and cDNA notation."
    pre    = "Review the following clinical genetics report.\n\n"
    hay    = _build_bio_haystack(rng, target_toks * 5, "clinical")
    for _ in range(20):
        split = int(len(hay) * depth)
        full  = pre + hay[:split] + "\n\n" + needle + "\n\n" + hay[split:] + q_text
        ids   = tok(full, return_tensors="pt").input_ids
        if ids.shape[1] <= target_toks + 200:
            break
        hay = hay[:int(len(hay) * 0.92)]
    return ids, answer


def build_bio_vt(rng: random.Random, tok, target_toks: int):
    pairs = rng.sample(DRUG_INTERACTION_PAIRS, min(CHAIN_LEN, len(DRUG_INTERACTION_PAIRS)))
    chain_sentences = []
    for drug_a, drug_b, mechanism in pairs:
        chain_sentences.append(
            f"Co-administration of {drug_a} with {drug_b} results in {mechanism}.")
    final_drug_a = pairs[0][0]
    answer = pairs[0][1]
    q_text = f"\n\nBased on the drug interaction data, what drug is directly affected by co-administration with {final_drug_a}? Answer with only the drug name."
    pre    = "Review the following clinical pharmacology notes.\n\n"
    hay    = _build_bio_haystack(rng, target_toks * 5, "clinical")
    chain_text = " ".join(chain_sentences)
    for _ in range(20):
        full = pre + hay[:int(len(hay) * 0.3)] + "\n\n" + chain_text + "\n\n" + hay[int(len(hay) * 0.3):] + q_text
        ids  = tok(full, return_tensors="pt").input_ids
        if ids.shape[1] <= target_toks + 200:
            break
        hay = hay[:int(len(hay) * 0.92)]
    return ids, answer


def build_clinical_niah(rng: random.Random, tok, target_toks: int, depth: float):
    icd_code, description = rng.choice(ICD_POOL)
    needle = (f"The patient's primary diagnosis was coded as {icd_code} "
              f"({description}) in the discharge summary.")
    answer = icd_code
    q_text = "\n\nWhat ICD-10 code was assigned as the primary diagnosis? Answer with only the code."
    pre    = "Review the following clinical documentation.\n\n"
    hay    = _build_bio_haystack(rng, target_toks * 5, "clinical")
    for _ in range(20):
        split = int(len(hay) * depth)
        full  = pre + hay[:split] + "\n\n" + needle + "\n\n" + hay[split:] + q_text
        ids   = tok(full, return_tensors="pt").input_ids
        if ids.shape[1] <= target_toks + 200:
            break
        hay = hay[:int(len(hay) * 0.92)]
    return ids, answer


_LB_CACHE: dict = {}

def _load_hf_dataset(name: str, config: str, split: str, n: int) -> list:
    key = f"{name}/{config}/{split}"
    if key in _LB_CACHE:
        return _LB_CACHE[key][:n]
    from datasets import load_dataset
    print(f"  Loading HuggingFace dataset: {name} [{config}] {split}...", flush=True)
    ds = load_dataset(name, config, split=split, trust_remote_code=True)
    samples = list(ds)
    _LB_CACHE[key] = samples
    return samples[:n]


def load_bio_lb_task(task: str, n: int) -> list:
    if task == "pubmedqa":
        raw = _load_hf_dataset("qiaojin/PubMedQA", "pqa_labeled", "train", n * 3)
        out = []
        for s in raw:
            ctx_parts = s.get("context", {}).get("contexts", [])
            context   = " ".join(ctx_parts) if ctx_parts else ""
            out.append({
                "context":  context,
                "question": s.get("question", ""),
                "answer":   s.get("final_decision", ""),
            })
            if len(out) >= n:
                break
        return out

    elif task == "pubmedqa_long":
        raw = _load_hf_dataset("qiaojin/PubMedQA", "pqa_unlabeled", "train", n * 3)
        out = []
        for s in raw:
            ctx_parts = s.get("context", {}).get("contexts", [])
            context   = " ".join(ctx_parts) if ctx_parts else ""
            long_ans  = s.get("long_answer", "")
            if not long_ans or not context:
                continue
            out.append({
                "context":  context,
                "question": s.get("question", ""),
                "answer":   long_ans[:200],
            })
            if len(out) >= n:
                break
        return out

    elif task == "medmcqa":
        raw = _load_hf_dataset("medmcqa", "default", "train", n * 3)
        choice_keys = ["opa", "opb", "opc", "opd"]
        out = []
        for s in raw:
            choices = [s.get(k, "") for k in choice_keys]
            correct_idx = s.get("cop", 0)
            context = (f"Question context: {s.get('exp', '')} "
                       + " ".join(f"Option {chr(65+i)}: {c}" for i, c in enumerate(choices)))
            answer = choices[correct_idx] if correct_idx < len(choices) else ""
            if not answer:
                continue
            out.append({
                "context":  context,
                "question": s.get("question", ""),
                "answer":   answer,
            })
            if len(out) >= n:
                break
        return out

    elif task == "medalpaca_medqa":
        raw = _load_hf_dataset("medalpaca/medical_meadow_medqa", "default", "train", n * 3)
        out = []
        for s in raw:
            out.append({
                "context":  s.get("input", ""),
                "question": s.get("instruction", ""),
                "answer":   s.get("output", ""),
            })
            if len(out) >= n:
                break
        return out

    elif task == "medalpaca_wiki":
        raw = _load_hf_dataset("medalpaca/medical_meadow_wikidoc", "default", "train", n * 3)
        out = []
        for s in raw:
            out.append({
                "context":  s.get("input", ""),
                "question": s.get("instruction", ""),
                "answer":   s.get("output", ""),
            })
            if len(out) >= n:
                break
        return out

    elif task == "clinical_niah":
        rng = random.Random(SEED + 99)
        out = []
        for _ in range(n * 2):
            icd_code, desc = rng.choice(ICD_POOL)
            hay = _build_bio_haystack(rng, 2000, "clinical")
            out.append({
                "context":  hay,
                "question": "What ICD-10 diagnosis code appears in the clinical note?",
                "needle":   icd_code,
                "answer":   icd_code,
                "_needle_sentence": f"Primary diagnosis: {icd_code} ({desc}).",
            })
            if len(out) >= n:
                break
        return out

    else:
        raise ValueError(f"Unknown bio LB task: {task}")


BIO_LB_PROMPTS: dict = {
    "pubmedqa": (
        "You are a biomedical research assistant. Based on the following study context, "
        "answer the question with yes, no, or maybe.\n\n"
        "Study Context:\n{context}\n\nQuestion: {question}\n\nAnswer (yes/no/maybe):"
    ),
    "pubmedqa_long": (
        "You are a biomedical research assistant. Based on the following study, "
        "provide a concise answer to the question.\n\n"
        "Study:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    ),
    "medmcqa": (
        "You are a medical expert. Answer the following medical question based on the context.\n\n"
        "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    ),
    "medalpaca_medqa": (
        "You are a medical expert. {question}\n\nContext: {context}\n\nAnswer:"
    ),
    "medalpaca_wiki": (
        "You are a medical expert. {question}\n\nContext: {context}\n\nAnswer:"
    ),
    "clinical_niah": (
        "Review the following clinical note and answer the question.\n\n"
        "Clinical Note:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    ),
}


def build_bio_lb_ids(sample: dict, task: str, tok, max_ctx: int) -> torch.Tensor:
    tmpl    = BIO_LB_PROMPTS[task]
    context = sample.get("context", "")
    question = sample.get("question", "")

    if task == "clinical_niah" and "_needle_sentence" in sample:
        needle_s = sample["_needle_sentence"]
        mid = len(context) // 2
        context = context[:mid] + " " + needle_s + " " + context[mid:]

    full = tmpl.format(context=context, question=question)
    ids  = tok(full, return_tensors="pt").input_ids

    if ids.shape[1] > max_ctx:
        half    = int(len(context) * (max_ctx / max(ids.shape[1], 1)) * 0.85) // 2
        context = context[:half] + context[len(context) - half:]
        full    = tmpl.format(context=context, question=question)
        ids     = tok(full, return_tensors="pt").input_ids
    return ids


def get_bio_lb_answers(sample: dict) -> list:
    ans = sample.get("answer", sample.get("answers", ""))
    if isinstance(ans, list):
        return ans
    return [str(ans)]


def ckpt_key(source: str, task: str, ctx: int, idx: int) -> str:
    return f"{source}_{task}_ctx{ctx}_i{idx:04d}"

def load_checkpoints() -> dict:
    done: dict = {}
    for f in CKPT_DIR.glob("*.json"):
        done[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    return done

def save_checkpoint(key: str, data: dict) -> None:
    (CKPT_DIR / f"{key}.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8")

def _init_csv(path: Path, cols: list) -> None:
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=cols).writeheader()

def _append_csv(path: Path, cols: list, rows: list) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        for r in rows:
            w.writerow(r)


def _empty_scores() -> dict:
    return {m: [] for m in METRIC_KEYS}

def _make_results_skeleton(pol_names: list) -> dict:
    res: dict = {}
    for src in ("bio_ruler", "bio_lb"):
        res[src] = {}
        tasks = BIO_RULER_TASKS if src == "bio_ruler" else BIO_LB_TASKS
        for t in tasks:
            res[src][t] = {}
            for ctx in CTX_LENS:
                res[src][t][ctx] = {}
                for pol in pol_names:
                    res[src][t][ctx][pol] = {B: _empty_scores() for B in BUDGETS}
    return res

def _reload_results_from_checkpoints(completed: dict, results: dict, pol_names: list) -> None:
    for key, data in completed.items():
        try:
            parts = key.split("_ctx")
            pre, rest = parts[0], parts[1]
            ctx_s, _ = rest.split("_i")
            ctx = int(ctx_s)
            src_task = pre.split("_", 1)
            source = src_task[0]
            task   = src_task[1]
            if source not in results or task not in results[source]:
                continue
            if ctx not in results[source][task]:
                continue
            for pol in pol_names:
                for B in BUDGETS:
                    for m in METRIC_KEYS:
                        v = data.get(pol, {}).get(str(B), {}).get(m)
                        if v is not None:
                            results[source][task][ctx][pol][B][m].append(v)
        except Exception:
            pass


def run_trial(
    source: str, task: str, ctx: int, trial_idx: int,
    ids: torch.Tensor, ground_truth, model, tok,
    results: dict, pol_names: list,
) -> tuple:
    device  = next(model.parameters()).device
    ids     = ids.to(device)
    seq_len = ids.shape[1]

    trial_data: dict  = {}
    pred_rows:  list  = []
    speed_rows: list  = []
    coherence_rows: list = []

    _vram_reset()
    t_sal0 = time.perf_counter()
    try:
        sals = extract_all_saliency(ids, model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM saliency ctx={ctx} trial={trial_idx}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return trial_data, pred_rows, speed_rows, coherence_rows
    sal_ms   = (time.perf_counter() - t_sal0) * 1000
    vram_sal = _vram_peak_mb()

    _vram_reset()
    t0 = time.perf_counter()
    try:
        pred_fc = gen_full(model, tok, ids)
    except Exception as e:
        pred_fc = ""
        print(f"  FullContext gen error: {e}", flush=True)
    gen_ms_fc   = (time.perf_counter() - t0) * 1000
    vram_gen_fc = _vram_peak_mb()
    new_toks_fc = len(tok.encode(pred_fc)) if pred_fc else 0
    tps_fc      = new_toks_fc / max(gen_ms_fc / 1000, 1e-6)
    mets_fc     = compute_metrics(pred_fc, ground_truth)

    trial_data["FullContext"] = {str(B): mets_fc for B in BUDGETS}
    for B in BUDGETS:
        for m in METRIC_KEYS:
            results[source][task][ctx]["FullContext"][B][m].append(mets_fc[m])
        pred_rows.append({
            "source": source, "task": task, "ctx": ctx,
            "trial_or_sample": trial_idx, "policy": "FullContext", "budget": B,
            "ground_truth": str(ground_truth), "prediction": pred_fc,
            **mets_fc, "llm_judge_score": "", "llm_judge_reason": "",
        })
        speed_rows.append({
            "source": source, "task": task, "ctx": ctx,
            "trial_or_sample": trial_idx, "policy": "FullContext", "budget": B,
            "sal_ms": round(sal_ms, 1), "gen_ms": round(gen_ms_fc, 1),
            "tokens_per_sec": round(tps_fc, 2),
            "vram_sal_mb": round(vram_sal, 1), "vram_gen_mb": round(vram_gen_fc, 1),
        })
        coherence_rows.append({
            "source": source, "task": task, "ctx": ctx,
            "trial_or_sample": trial_idx, "policy": "FullContext", "budget": B,
            "eviction_coherence_loss": measure_coherence_loss(model, ids, set(range(seq_len))),
        })

    for pol_name, (sig_key, pol_fn) in POLICIES.items():
        trial_data[pol_name] = {}
        for B in BUDGETS:
            try:
                if B >= seq_len:
                    pred = gen_full(model, tok, ids)
                    keep = set(range(seq_len))
                    gen_ms = gen_ms_fc; vram_gen = vram_gen_fc
                else:
                    keep = pol_fn(sals, B, seq_len)
                    _vram_reset()
                    t_g0 = time.perf_counter()
                    pred = gen_evict(model, tok, ids, keep)
                    gen_ms = (time.perf_counter() - t_g0) * 1000
                    vram_gen = _vram_peak_mb()

                new_toks = len(tok.encode(pred)) if pred else 0
                tps      = new_toks / max(gen_ms / 1000, 1e-6) if B < seq_len else tps_fc
                mets     = compute_metrics(pred, ground_truth)
                coh_val  = measure_coherence_loss(model, ids, keep)

                trial_data[pol_name][str(B)] = mets
                for m in METRIC_KEYS:
                    results[source][task][ctx][pol_name][B][m].append(mets[m])

                pred_rows.append({
                    "source": source, "task": task, "ctx": ctx,
                    "trial_or_sample": trial_idx, "policy": pol_name, "budget": B,
                    "ground_truth": str(ground_truth), "prediction": pred,
                    **mets, "llm_judge_score": "", "llm_judge_reason": "",
                })
                speed_rows.append({
                    "source": source, "task": task, "ctx": ctx,
                    "trial_or_sample": trial_idx, "policy": pol_name, "budget": B,
                    "sal_ms": round(sal_ms, 1),
                    "gen_ms": round(gen_ms if B < seq_len else gen_ms_fc, 1),
                    "tokens_per_sec": round(tps, 2),
                    "vram_sal_mb": round(vram_sal, 1),
                    "vram_gen_mb": round(vram_gen if B < seq_len else vram_gen_fc, 1),
                })
                coherence_rows.append({
                    "source": source, "task": task, "ctx": ctx,
                    "trial_or_sample": trial_idx, "policy": pol_name, "budget": B,
                    "eviction_coherence_loss": round(coh_val, 4),
                })

            except torch.cuda.OutOfMemoryError:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  OOM {pol_name} B={B}", flush=True)
            except Exception as e:
                print(f"  {pol_name} B={B} error: {e}", flush=True)

    return trial_data, pred_rows, speed_rows, coherence_rows


def main() -> None:
    _init_csv(PRED_CSV_PATH,      PRED_COLS)
    _init_csv(SPEED_CSV_PATH,     SPEED_COLS)
    _init_csv(COHERENCE_CSV_PATH, COHERENCE_COLS)

    completed = load_checkpoints()
    print(f"Resuming: {len(completed)} trials already done.", flush=True)

    model, tok = load_model()

    pol_names = list(POLICIES) + ["FullContext"]
    results   = _make_results_skeleton(pol_names)
    _reload_results_from_checkpoints(completed, results, pol_names)

    t0_total = time.time()
    done     = len(completed)
    total_ruler = len(BIO_RULER_TASKS) * len(CTX_LENS) * N_TRIALS
    total_lb    = len(BIO_LB_TASKS)   * len(CTX_LENS) * LB_SAMPLES
    total       = total_ruler + total_lb

    ruler_depths_cycle = [0.25, 0.5, 0.75]
    for task_name in BIO_RULER_TASKS:
        for ctx_len in CTX_LENS:
            print(f"\n{'='*60}", flush=True)
            print(f"[Bio-RULER] {task_name} @ ctx={ctx_len}", flush=True)
            rng_task = random.Random(ctx_len * 7 + hash(task_name) % 1000)

            for trial in range(N_TRIALS):
                key = ckpt_key("bio_ruler", task_name, ctx_len, trial)
                if key in completed:
                    continue

                rng_t = random.Random(trial * 31337 + ctx_len + SEED)
                depth = rng_task.choice(ruler_depths_cycle)
                try:
                    if task_name == "bio_niah_single":
                        ids, gt = build_bio_niah_single(rng_t, tok, ctx_len, depth)
                    elif task_name == "bio_niah_gene":
                        ids, gt = build_bio_niah_gene(rng_t, tok, ctx_len, depth)
                    elif task_name == "bio_vt":
                        ids, gt = build_bio_vt(rng_t, tok, ctx_len)
                    else:
                        ids, gt = build_clinical_niah(rng_t, tok, ctx_len, depth)
                except Exception as e:
                    print(f"  build error trial={trial}: {e}", flush=True)
                    continue

                trial_data, pr, sr, cr = run_trial(
                    "bio_ruler", task_name, ctx_len, trial,
                    ids, gt, model, tok, results, pol_names)

                save_checkpoint(key, trial_data)
                _append_csv(PRED_CSV_PATH,      PRED_COLS,      pr)
                _append_csv(SPEED_CSV_PATH,     SPEED_COLS,     sr)
                _append_csv(COHERENCE_CSV_PATH, COHERENCE_COLS, cr)
                done += 1

                if done % 5 == 0:
                    elapsed = (time.time() - t0_total) / 60
                    ki = np.mean(results["bio_ruler"][task_name][ctx_len]["KiaOmni_s8"][256]["contains"] or [0])
                    sk = np.mean(results["bio_ruler"][task_name][ctx_len]["SnapKV_Modified"][256]["contains"] or [0])
                    print(f"  [{done}/{total}] {elapsed:.1f}min | KiaOmni_s8={ki:.3f} SnapKV_Modified={sk:.3f}", flush=True)

    for task_name in BIO_LB_TASKS:
        try:
            samples = load_bio_lb_task(task_name, LB_SAMPLES)
        except Exception as e:
            print(f"  Skipping {task_name}: {e}", flush=True)
            continue

        for ctx_len in CTX_LENS:
            print(f"\n{'='*60}", flush=True)
            print(f"[Bio-LB] {task_name} @ ctx={ctx_len}", flush=True)

            for si, sample in enumerate(samples):
                key = ckpt_key("bio_lb", task_name, ctx_len, si)
                if key in completed:
                    continue

                try:
                    ids = build_bio_lb_ids(sample, task_name, tok, ctx_len)
                    gt  = get_bio_lb_answers(sample)
                except Exception as e:
                    print(f"  build error sample={si}: {e}", flush=True)
                    continue

                trial_data, pr, sr, cr = run_trial(
                    "bio_lb", task_name, ctx_len, si,
                    ids, gt, model, tok, results, pol_names)

                save_checkpoint(key, trial_data)
                _append_csv(PRED_CSV_PATH,      PRED_COLS,      pr)
                _append_csv(SPEED_CSV_PATH,     SPEED_COLS,     sr)
                _append_csv(COHERENCE_CSV_PATH, COHERENCE_COLS, cr)
                done += 1

                if done % 5 == 0:
                    elapsed = (time.time() - t0_total) / 60
                    print(f"  [{done}/{total}] {elapsed:.1f}min", flush=True)

    summary: dict = {}
    for src in ("bio_ruler", "bio_lb"):
        summary[src] = {}
        tasks = BIO_RULER_TASKS if src == "bio_ruler" else BIO_LB_TASKS
        for t in tasks:
            summary[src][t] = {}
            for ctx in CTX_LENS:
                summary[src][t][ctx] = {}
                for pol in pol_names:
                    summary[src][t][ctx][pol] = {}
                    for B in BUDGETS:
                        sc = results[src][t][ctx][pol][B]
                        summary[src][t][ctx][pol][B] = {
                            m: (float(np.mean(sc[m])) if sc[m] else None)
                            for m in METRIC_KEYS
                        }
                        summary[src][t][ctx][pol][B]["n"] = len(sc["f1"])

    macro: dict = {pol: {B: [] for B in BUDGETS} for pol in pol_names}
    for src in ("bio_ruler", "bio_lb"):
        tasks = BIO_RULER_TASKS if src == "bio_ruler" else BIO_LB_TASKS
        for t in tasks:
            for ctx in CTX_LENS:
                for pol in pol_names:
                    for B in BUDGETS:
                        v = summary[src][t][ctx][pol][B]["contains"]
                        if v is not None:
                            macro[pol][B].append(v)

    macro_avg = {
        pol: {B: (float(np.mean(macro[pol][B])) if macro[pol][B] else None)
              for B in BUDGETS}
        for pol in pol_names
    }

    print(f"\n{'='*60}", flush=True)
    print("[038] MACRO AVG CONTAINS (bio_niah focus - all tasks + contexts):", flush=True)
    for pol, bdict in macro_avg.items():
        row = "  ".join(f"B={B}:{v:.3f}" for B, v in bdict.items() if v is not None)
        print(f"  {pol:<22}  {row}", flush=True)

    out = {
        "experiment":    "038_biomistral_comparison",
        "model":         MODEL_NAME,
        "policies":      list(POLICIES.keys()) + ["FullContext"],
        "bio_ruler_tasks": BIO_RULER_TASKS,
        "bio_lb_tasks":  BIO_LB_TASKS,
        "ctx_lens":      CTX_LENS,
        "budgets":       BUDGETS,
        "n_trials_ruler": N_TRIALS,
        "n_samples_lb":  LB_SAMPLES,
        "macro_avg_contains": macro_avg,
        "per_source_task_ctx": summary,
        "paper_claim":   (
            "KiaOmni sigma=8 preserves multi-subword clinical identifiers "
            "(rsIDs, HGVS, ICD codes) better than pointwise eviction methods "
            "because boxcar smoothing fills intra-identifier subword gaps."
        ),
    }
    rpath = OUT_DIR / "results.json"
    rpath.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults     -> {rpath}", flush=True)
    print(f"Predictions -> {PRED_CSV_PATH}", flush=True)
    print(f"Speed/VRAM  -> {SPEED_CSV_PATH}", flush=True)
    print(f"Coherence   -> {COHERENCE_CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
