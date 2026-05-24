translations = {
    # ============================
    # SIDEBAR
    # ============================
    "sidebar_title": {"ro": "⚙️ Controale", "en": "⚙️ Controls"},
    "sidebar_language_selector": {"ro": "Limbă", "en": "Language"},
    "sidebar_method": {"ro": "Metodă sentiment", "en": "Sentiment method"},
    "sidebar_company": {"ro": "Selectează compania", "en": "Select company"},
    "sidebar_rows_slider": {"ro": "Număr rânduri afișate", "en": "Rows to display"},
    "sidebar_limit_rows": {"ro": "Rânduri tabel dispute", "en": "Table rows limit"},
    "sidebar_refresh": {"ro": "🔄 Reîmprospătează Dashboard", "en": "🔄 Refresh Dashboard"},
    "sidebar_caption": {
        "ro": "Alege un tab de sus și ajustează controalele din stânga.",
        "en": "Choose a top tab and adjust controls on the left.",
    },

    # ============================
    # TAB TITLES
    # ============================
    "tab_dashboard": {"ro": "📊 Panou", "en": "📊 Dashboard"},
    "tab_ai_insights": {"ro": "🤖 Insight-uri AI", "en": "🤖 AI Insights"},
    "tab_interactive_demo": {"ro": "🧪 Demo Interactiv", "en": "🧪 Interactive Demo"},
    "tab_live_pipeline": {"ro": "🔄 Pipeline Live", "en": "🔄 Live Pipeline"},
    "tab_proof_source": {"ro": "📁 Dovada Sursei", "en": "📁 Proof of Source"},


    # ============================
    # DASHBOARD (TAB 1)
    # ============================
    "dashboard_intro_title": {"ro": "Ce face această aplicație?", "en": "What does this app do?"},
    "dashboard_intro_description": {
        "ro": "Această aplicație analizează discuțiile publice de pe Reddit despre companii tehnologice majore și detectează automat dacă sentimentul este pozitiv, neutru sau negativ.\n\nEa compară trei abordări diferite:\n\n- Logistic Regression (model machine learning de bază)\n- VADER (model bazat pe reguli)\n- Deep Learning Transformer\n\nFolosește controalele din stânga pentru a explora cum variază sentimentul în funcție de companie și metoda selectată.",
        
        "en": "This application analyzes public Reddit discussions about major tech companies and automatically determines whether the sentiment is positive, neutral, or negative.\n\nIt compares three different approaches:\n\n- Logistic Regression (machine learning baseline)\n- VADER (rule-based)\n- Deep Learning Transformer\n\nUse the controls on the left to explore how sentiment changes across companies and methods."
    },
"dashboard_overview_description": {
    "ro": "Setul de date conține postări și comentarii Reddit colectate prin API-ul public JSON al platformei. Procesul de colectare a folosit subreddit-uri din zona tehnologiei, filtrare pe cuvinte cheie, paginare, endpoint-uri de căutare și extragerea comentariilor. Mențiunile colectate sunt stocate într-o bază de date PostgreSQL și analizate folosind trei metode de analiză a sentimentului.",
    "en": "The dataset contains Reddit posts and comments collected through the public Reddit JSON API. The collection process used technology-related subreddits, keyword filtering, pagination, search endpoints, and comment extraction. The collected mentions are stored in a PostgreSQL database and analyzed using three sentiment analysis methods."
},
"dashboard_overview_note": {
    "ro": "Notă: Media încrederii reprezintă probabilitatea medie a modelului selectat. Pentru VADER, scorul reprezintă valoarea compound a sentimentului.\n\nDashboard-ul folosește datele aflate în prezent în baza de date PostgreSQL.",
    "en": "Note: Avg confidence represents the average confidence/probability of the selected model. For VADER, the score represents the compound sentiment score.\n\nThe dashboard uses the data currently stored in the PostgreSQL database."
},

    "dashboard_overview_title": {"ro": "Prezentare generală", "en": "Overview of collected data"},
    "dashboard_total_mentions": {"ro": "Mențiuni totale", "en": "Total mentions"},
    "dashboard_avg_score": {"ro": "Scor mediu", "en": "Average score"},
    "dashboard_pct_negative": {"ro": "Procent negativ", "en": "Percent negative"},
    "dashboard_pct_disagreement": {"ro": "Dezacord modele", "en": "Model disagreement"},
    "dashboard_pct_disagreement_vader": {"ro": "Dezacord LR vs VADER", "en": "LR vs VADER disagreement"},
    "dashboard_pct_disagreement_dl": {"ro": "Dezacord LR vs DL", "en": "LR vs DL disagreement"},
    "dashboard_sentiment_distribution_title": {"ro": "Distribuția sentimentelor funcție de metodă", "en": "Sentiment distribution by method"},
    "dashboard_no_data_available": {"ro": "Nu există date disponibile.", "en": "No data available."},
    "dashboard_sentiment_breakdown_title": {"ro": "Detaliere sentimente", "en": "Sentiment breakdown"},
    "dashboard_positive_label": {"ro": "Pozitive", "en": "Positive"},
    "dashboard_negative_label": {"ro": "Negative", "en": "Negative"},
    "dashboard_neutral_label": {"ro": "Neutre", "en": "Neutral"},
    "dashboard_company_summary_title": {"ro": "Sumar pe companie", "en": "Company summary"},
    "dashboard_download_summary_button": {"ro": "Descarcă sumarul", "en": "Download summary"},
    "dashboard_charts_title": {"ro": "Grafice", "en": "Charts"},
    "dashboard_most_negative_mentions_title": {"ro": "Cele mai negative mențiuni", "en": "Most negative mentions"},
    "dashboard_no_negative_examples": {"ro": "Nu există exemple negative.", "en": "No negative examples available."},
    "dashboard_model_evaluation_title": {"ro": "Evaluarea modelului", "en": "Model evaluation"},
    "dashboard_raw_confusion_matrix_title": {"ro": "Matrice de confuzie (brută)", "en": "Raw confusion matrix"},
    "dashboard_normalized_confusion_matrix_title": {"ro": "Matrice de confuzie (normalizată)", "en": "Normalized confusion matrix"},
    "dashboard_confusion_matrix_not_numeric": {
        "ro": "Matricea de confuzie nu conține valori numerice.",
        "en": "Confusion matrix does not contain numeric values.",
    },
    "dashboard_classification_report_title": {"ro": "Raport de clasificare", "en": "Classification report"},
    "dashboard_manual_validation_title": {"ro": "Validare manuală", "en": "Manual validation"},
    "dashboard_sample_size_label": {"ro": "Dimensiune eșantion", "en": "Sample size"},
    "dashboard_accuracy_label": {"ro": "Acuratețe", "en": "Accuracy"},
    "dashboard_f1_macro_label": {"ro": "F1 macro", "en": "F1 macro"},
    "dashboard_manual_confusion_matrix_not_numeric": {
        "ro": "Matricea manuală nu conține valori numerice.",
        "en": "Manual confusion matrix does not contain numeric values.",
    },
    "dashboard_no_manual_confusion_matrix": {
        "ro": "Nu există matrice de confuzie manuală.",
        "en": "No manual confusion matrix available.",
    },
    "dashboard_manual_classification_report_title": {
        "ro": "Raport manual de clasificare",
        "en": "Manual classification report",
    },
    "dashboard_manual_sample_title": {"ro": "Eșantion manual", "en": "Manual sample"},
    "dashboard_no_manual_sample_data": {"ro": "Nu există date de eșantion manual.", "en": "No manual sample data available."},
    "dashboard_no_manual_validation_method": {
        "ro": "Nu există validare manuală pentru această metodă.",
        "en": "No manual validation available for this method.",
    },

    # ============================
    # TAB 3 — INTERACTIVE DEMO
    # ============================
    "tab3_title": {"ro": "🧪 Demo Interactiv", "en": "🧪 Interactive Model Demo"},
    "tab3_intro_description": {
        "ro": "Compară cum trei modele diferite de analiză a sentimentului interpretează același text.",
        "en": "Compare how three different sentiment analysis models interpret the same text.",
    },
    "tab3_input_section": {"ro": "✍️ Text de intrare", "en": "✍️ Input text"},
    "tab3_choose_example": {"ro": "Alege o propoziție de exemplu", "en": "Choose an example sentence"},
    "tab3_write_custom": {"ro": "Sau scrie propria ta propoziție:", "en": "Or write your own sentence:"},
    "tab3_analyze_button": {"ro": "Analizează textul", "en": "Analyze text"},
    "tab3_empty_text_warning": {"ro": "Te rog introduceți un text mai întâi.", "en": "Please enter a text first."},
    "tab3_results_section": {"ro": "📊 Rezultatele modelelor", "en": "📊 Model results"},
    "tab3_logreg_title": {"ro": "🔵 Logistic Regression", "en": "🔵 Logistic Regression"},
    "tab3_prediction_label": {"ro": "Predicție:", "en": "Prediction:"},
    "tab3_logreg_word_contrib_title": {"ro": "Contribuții cuvinte (LR)", "en": "Word Contributions (LR)"},
    "tab3_logreg_word_contrib_subtitle": {"ro": "TF-IDF × LR greutate", "en": "TF-IDF × LR weight"},
    "tab3_vader_title": {"ro": "🟢 VADER", "en": "🟢 VADER"},
    "tab3_vader_word_contrib_title": {"ro": "Contribuții cuvinte (VADER)", "en": "Word Contributions (VADER)"},
    "tab3_vader_word_contrib_subtitle": {"ro": "Scor lexicon VADER", "en": "VADER lexicon score"},
    "tab3_transformer_title": {"ro": "🧠 Transformer", "en": "🧠 Transformer"},
    "tab3_longer_sentence_info": {
        "ro": "Scrie o propoziție mai lungă pentru a afișa heatmap-ul de atenție.",
        "en": "Write a longer sentence to display the attention heatmap.",
    },
    "tab3_comparison_section": {"ro": "📌 Comparație finală", "en": "📌 Final comparison"},
    "tab3_comparison_success": {
        "ro": "Acest demo te ajută să înțelegi cum diferitele modele NLP interpretează același text.",
        "en": "This demo helps you understand how different NLP models interpret the same text.",
    },

    # ============================
    # TAB 4 — LIVE PIPELINE
    # ============================
    "tab4_title": {"ro": "Live Reddit Pipeline Demo", "en": "Live Reddit Pipeline Demo"},
    "tab4_description": {
        "ro": "Acest demo reproduce întregul pipeline în 5 pași.",
        "en": "This demo reproduces the full 5-step pipeline.",
    },
    "tab4_demo_mode_caption": {
        "ro": "DEMO MODE ACTIV — limitat la 60 comentarii.",
        "en": "DEMO MODE ACTIVE — limited to 60 comments.",
    },
    "tab4_comments_per_company": {"ro": "Comentarii per companie", "en": "Comments per company"},
    "tab4_expected_sample_caption": {
        "ro": "Dimensiune estimată: {count} comentarii.",
        "en": "Estimated size: {count} comments.",
    },
    "tab4_run_pipeline_title": {"ro": "Rulează pipeline-ul", "en": "Run pipeline"},
    "tab4_run_button": {"ro": "Rulează pipeline-ul Reddit live", "en": "Run live Reddit pipeline"},
    "tab4_press_button_info": {
        "ro": "Apasă butonul pentru a rula pipeline-ul.",
        "en": "Press the button to run the pipeline.",
    },
    "tab4_pipeline_steps_title": {"ro": "Pașii pipeline-ului", "en": "Pipeline steps"},
    "tab4_step_collect": {"ro": "Colectare", "en": "Collect"},
    "tab4_step_extract": {"ro": "Extracție", "en": "Extract"},
    "tab4_step_map": {"ro": "Mapare", "en": "Map"},
    "tab4_step_analyze": {"ro": "Analiză", "en": "Analyze"},
    "tab4_step_result": {"ro": "Rezultat", "en": "Result"},
    "tab4_results_title": {"ro": "Rezultatele pipeline-ului", "en": "Pipeline results"},
    "tab4_comments_metric": {"ro": "Comentarii", "en": "Comments"},
    "tab4_companies_metric": {"ro": "Companii", "en": "Companies"},
    "tab4_models_metric": {"ro": "Modele", "en": "Models"},
    "tab4_no_valid_timestamps_caption": {
        "ro": "Nu există timestamp-uri valide.",
        "en": "No valid timestamps available.",
    },

    # ============================
    # TAB 5 — PROOF OF SOURCE
    # ============================
    "tab5_title": {"ro": "📁 Dovada Sursei", "en": "📁 Proof of Source"},
    "tab5_description": {
        "ro": "Răsfoiește mențiunile brute colectate de pe Reddit.",
        "en": "Browse the raw Reddit mentions collected in the dataset.",
    },
    "tab5_rows_per_page": {"ro": "Rânduri per pagină", "en": "Rows per page"},
    "tab5_prev_button": {"ro": "⬅ Anterior", "en": "⬅ Previous"},
    "tab5_next_button": {"ro": "Următor ➡", "en": "Next ➡"},
    "tab5_page_label": {"ro": "Pagina {page} / {total}", "en": "Page {page} / {total}"},
    "tab5_showing_rows": {
        "ro": "Afișezi rândurile {start}–{end} din {total}",
        "en": "Showing rows {start}–{end} of {total}",
    },
    "tab5_select_row_title": {"ro": "Selectează un rând", "en": "Select a row"},
    "tab5_select_row_label": {"ro": "Alege un rând:", "en": "Choose a row:"},
    "tab5_inspect_title": {"ro": "🔎 Inspectează mențiunea selectată", "en": "🔎 Inspect selected mention"},

    # ============================
    # AI INSIGHTS (TAB 2)
    # ============================
    "ai_insights_header_title": {
        "ro": "AI Insights — Marketing & Reputation Intelligence",
        "en": "AI Insights — Marketing & Reputation Intelligence",
    },
    "ai_insights_header_subtitle": {
        "ro": "Interpretare automată a percepției publice pe baza comentariilor Reddit.",
        "en": "Automated interpretation of public perception based on Reddit comments.",
    },
    "ai_insights_context_selector": {"ro": "Selectează contextul analizei", "en": "Select analysis context"},
    "ai_insights_sentiment_method": {"ro": "Metodă sentiment", "en": "Sentiment method"},
    "ai_insights_company": {"ro": "Companie", "en": "Company"},
    "ai_insights_period": {"ro": "Perioadă", "en": "Period"},
    "ai_insights_llm_engine": {"ro": "LLM engine", "en": "LLM engine"},
    "ai_insights_which_insights": {"ro": "Ce insight-uri vrei să generezi?", "en": "Which insights do you want to generate?"},
    "ai_insights_executive_summary_checkbox": {"ro": "Rezumat executiv", "en": "Executive summary"},
    "ai_insights_general_trend_checkbox": {"ro": "Tendință generală", "en": "General trend"},
    "ai_insights_dominant_themes_checkbox": {"ro": "Teme dominante", "en": "Dominant themes"},
    "ai_insights_reputation_risks_checkbox": {"ro": "Riscuri reputaționale", "en": "Reputation risks"},
    "ai_insights_marketing_recs_checkbox": {"ro": "Recomandări de marketing", "en": "Marketing recommendations"},
    "ai_insights_bullet_mode_checkbox": {"ro": "Mod concis", "en": "Concise mode"},
    "ai_insights_detail_level": {"ro": "Nivel detaliu", "en": "Detail level"},
    "ai_insights_presentation_mode": {"ro": "Mod prezentare", "en": "Presentation mode"},
    "ai_insights_generate_button": {"ro": "🚀 Generează AI Insights", "en": "🚀 Generate AI Insights"},
    "ai_insights_no_insights_message": {"ro": "Apasă butonul pentru a genera insight-uri.", "en": "Press the button to generate insights."},
    "ai_insights_llm_trace_viewer": {"ro": "🔍 LLM Trace Viewer — Debug complet", "en": "🔍 LLM Trace Viewer — Full Debug"},
    "ai_insights_llm_context_header": {"ro": "📌 Context trimis către model", "en": "📌 Context sent to model"},
    "ai_insights_llm_raw_output_header": {"ro": "📌 Output brut", "en": "📌 Raw output"},
    "ai_insights_llm_execution_times_header": {"ro": "📌 Timpi de execuție", "en": "📌 Execution times"},
    "ai_insights_llm_chains_header": {"ro": "📌 Chain-uri rulate", "en": "📌 Chains executed"},
    "ai_insights_llm_prompts_note_header": {"ro": "📌 Notă despre prompturi", "en": "📌 Note about prompts"},
    "ai_insights_llm_prompts_default_note": {"ro": "Multiple prompts used.", "en": "Multiple prompts used."},
    "ai_insights_generating_spinner": {"ro": "Se generează insight-urile...", "en": "Generating insights..."},
    "ai_insights_rate_limit_warning": {
        "ro": "⚠️ Limită de tokeni atinsă. Se folosește modelul mic.",
        "en": "⚠️ Token limit reached. Using small model.",
    },
    "ai_insights_cache_loaded_info": {
        "ro": "Insight-urile au fost încărcate din cache.",
        "en": "Insights loaded from cache.",
    },
    "help_dialog_title": {
    "ro": "Asistent Dashboard Reputație",
    "en": "Reputation Dashboard Assistant",
    },
    "help_intro_1": {
        "ro": "👋 Salut! Îți pot explica modul în care funcționează această aplicație de analiză a reputației.",
        "en": "👋 Hi! I can explain how this reputation analysis app works.",
    },
    "help_intro_2": {
        "ro": "Întreabă-mă despre: **Dashboard**, **AI Insights**, **Logistic Regression**, **VADER**, **Transformer**, **Live Pipeline** sau **Proof of Source**.",
        "en": "Ask me about: **Dashboard**, **AI Insights**, **Logistic Regression**, **VADER**, **Transformer**, **Live Pipeline**, or **Proof of Source**.",
    },
    "help_input_label": {
        "ro": "Întreabă despre aplicație",
        "en": "Ask about the app",
    },
    "help_input_placeholder": {
        "ro": "Exemplu: Explică VADER",
        "en": "Example: Explain VADER",
    },
    "help_send_button": {
        "ro": "Trimite",
        "en": "Send",
    },
    "ai_insights_meta_model": {"ro": "Model:", "en": "Model:"},
    "ai_insights_meta_company": {"ro": "Companie:", "en": "Company:"},
    "ai_insights_meta_period": {"ro": "Perioadă:", "en": "Period:"},
    "ai_insights_meta_method": {"ro": "Metodă sentiment:", "en": "Sentiment method:"},
    "ai_insights_meta_detail": {"ro": "Nivel detaliu:", "en": "Detail level:"},
    "ai_insights_meta_bullet": {"ro": "Mod concis:", "en": "Concise mode:"},

    "yes": {"ro": "Da", "en": "Yes"},
    "no": {"ro": "Nu", "en": "No"},
"ai_insights_card_executive_summary": {
    "ro": "Rezumat executiv",
    "en": "Executive summary",
},
"ai_insights_card_general_sentiment": {
    "ro": "Tendință generală",
    "en": "General sentiment",
},
"ai_insights_card_recurring_themes": {
    "ro": "Teme dominante",
    "en": "Recurring themes",
},
"ai_insights_card_reputation_risks": {
    "ro": "Riscuri reputaționale",
    "en": "Reputation risks",
},
"ai_insights_card_marketing_recommendations": {
    "ro": "Recomandări de marketing",
    "en": "Marketing recommendations",
},
# ============================
# DASHBOARD – INTERPRETATION
# ============================
"dashboard_interpretation_title": {
    "en": "Interpretation",
    "ro": "Interpretare"
},
"dashboard_interpretation_no_data": {
    "en": "No interpretation can be generated because no data is available.",
    "ro": "Nu se poate genera o interpretare deoarece nu există date disponibile."
},
"dashboard_interpretation_high_negative": {
    "en": "A high percentage of negative sentiment ({neg}%) indicates significant reputational risks.",
    "ro": "Un procent ridicat de sentiment negativ ({neg}%) indică riscuri reputaționale semnificative."
},
"dashboard_interpretation_moderate_negative": {
    "en": "A moderate level of negative sentiment ({neg}%) suggests mixed public perception.",
    "ro": "Un nivel moderat de sentiment negativ ({neg}%) sugerează o percepție publică mixtă."
},
"dashboard_interpretation_low_negative": {
    "en": "A low percentage of negative sentiment ({neg}%) indicates generally positive public perception.",
    "ro": "Un procent scăzut de sentiment negativ ({neg}%) indică o percepție publică în general pozitivă."
},

# ============================
# DASHBOARD – METHOD NOTE
# ============================
"dashboard_method_note_title": {
    "en": "Method Note",
    "ro": "Notă despre metodă"
},
"dashboard_method_note_lr": {
    "en": "The Logistic Regression model is a machine learning baseline trained on a 3‑class sentiment dataset.",
    "ro": "Modelul Logistic Regression este un baseline de machine learning antrenat pe un set de date cu 3 clase de sentiment."
},
"dashboard_method_note_vader": {
    "en": "VADER is a rule‑based sentiment analyzer optimized for social media text.",
    "ro": "VADER este un analizor de sentiment bazat pe reguli, optimizat pentru textul din social media."
},
"dashboard_method_note_dl": {
    "en": "The Deep Learning Transformer model provides context‑aware sentiment predictions.",
    "ro": "Modelul Deep Learning Transformer oferă predicții de sentiment sensibile la context."
},

# ============================
# DASHBOARD – DISAGREEMENT
# ============================
"dashboard_disagreement_lr_vader": {
    "en": "Method Disagreement: Logistic Regression vs VADER",
    "ro": "Diferențe între metode: Logistic Regression vs VADER"
},
"dashboard_disagreement_lr_dl": {
    "en": "Method Disagreement: Logistic Regression vs Deep Learning Transformer",
    "ro": "Diferențe între metode: Logistic Regression vs Deep Learning Transformer"
},
"dashboard_disagreement_caption": {
    "en": "This table shows how often the selected method disagrees with Logistic Regression on the same posts.",
    "ro": "Acest tabel arată cât de des metoda selectată diferă de Logistic Regression pe aceleași postări."
},

# ============================
# DASHBOARD – POSTS DISAGREE
# ============================
"dashboard_posts_disagree_lr_vader": {
    "en": "Posts where Logistic Regression and VADER disagree",
    "ro": "Postări unde Logistic Regression și VADER diferă"
},
"dashboard_posts_disagree_lr_dl": {
    "en": "Posts where Logistic Regression and Deep Learning Transformer disagree",
    "ro": "Postări unde Logistic Regression și Deep Learning Transformer diferă"
},
"dashboard_posts_disagree_caption": {
    "en": "These examples highlight posts where the two methods produce different sentiment labels.",
    "ro": "Aceste exemple evidențiază postările unde cele două metode produc etichete de sentiment diferite."
},
"help_title": {
    "ro": "Asistentul tău",
    "en": "Your Assistant",
},
"help_select_topic": {
    "ro": "Alege un subiect pentru explicații",
    "en": "Choose a topic for explanation",
},
"help_topic_label": {
    "ro": "Subiect",
    "en": "Topic",
},
"help_explanation_label": {
    "ro": "Explicație",
    "en": "Explanation",
},
"help_send_button": {
    "ro": "Trimite",
    "en": "Send",
},
"help_placeholder": {
    "ro": "Exemplu: VADER, Logistic Regression, Dashboard",
    "en": "Example: VADER, Logistic Regression, Dashboard",
},

}


def t(key, lang):
    return translations.get(key, {}).get(lang.lower(), translations.get(key, {}).get(lang, key))
