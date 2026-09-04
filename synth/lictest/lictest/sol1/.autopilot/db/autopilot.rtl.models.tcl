set SynModuleInfo {
  {SRCNAME topf MODELNAME topf RTLNAME topf IS_TOP 1
    SUBMODULES {
      {MODELNAME topf_mul_16s_16s_26_1_1 RTLNAME topf_mul_16s_16s_26_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME topf_mac_muladd_16s_16s_26ns_26_4_1 RTLNAME topf_mac_muladd_16s_16s_26ns_26_4_1 BINDTYPE op TYPE all IMPL dsp_slice LATENCY 3}
    }
  }
}
